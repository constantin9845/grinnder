#include "async_transfer.h"

#include <ATen/cuda/CUDAContext.h>

#include "../thread_pool.h"

#include <fcntl.h>
#include <unistd.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <cstring>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h> 
#include <cuda_runtime.h>
#include <cufile.h>
#include <future>
#include <cstdlib>
#include <mutex>
#include <algorithm>
#include <cstdint>
#include <cerrno>
#include <sys/ioctl.h>
#include <linux/nvme_ioctl.h>
#include <cstdio>
#include <ctime>
#include <atomic>
#include <thread>
#include <functional>

// Set GRINNDER_GDS_DIRECT=1 when cuFile runs on the real GDS path (nvidia-fs
// loaded, compat mode off). It turns on two workarounds that only that path needs:
//  - each batch IO gets its own devPtr_base with devPtr_offset = 0, because for
//    unregistered buffers GDS rejects devPtr_offset + size > 1 MiB;
//  - only one cuFile batch runs at a time, because overlapping batches hit an
//    internal libcufile assertion (cufio_batch line 521).
// Leave it unset for compat mode, which needs neither and runs faster without them.
static bool gds_direct_mode() {
    static const bool on = [] {
        const char* v = std::getenv("GRINNDER_GDS_DIRECT");
        return v && v[0] == '1';
    }();
    return on;
}

static std::unordered_map<int, CUfileHandle_t> g_cufile_handles;
static std::mutex g_cufile_handles_mutex;

static CUfileHandle_t get_cufile_handle(int fd) {
  std::lock_guard<std::mutex> lock(g_cufile_handles_mutex);
  auto it = g_cufile_handles.find(fd);
  AT_ASSERTM(it != g_cufile_handles.end(), 
             "File descriptor " + std::to_string(fd) + " not registered in g_cufile_handles");
  return it->second;
}

std::vector<int> open_files(const std::vector<std::string>& file_paths) {
  std::vector<int> fds;
  fds.reserve(file_paths.size());

  for (const auto& path : file_paths) {
    int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
    AT_ASSERTM(fd >= 0, "Failed to open file with O_DIRECT: " + path);

    CUfileDescr_t desc;
    memset(&desc, 0, sizeof(CUfileDescr_t));
    desc.handle.fd = fd;
    desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    CUfileHandle_t handle;
    CUfileError_t status = cuFileHandleRegister(&handle, &desc);
    AT_ASSERTM(status.err == CU_FILE_SUCCESS, 
               "cuFileHandleRegister failed for file: " + path);

    {
      std::lock_guard<std::mutex> lock(g_cufile_handles_mutex);
      g_cufile_handles[fd] = handle;
    }
    fds.push_back(fd);
  }
  return fds;
}

void close_files(const std::vector<int>& fds) {
  std::lock_guard<std::mutex> lock(g_cufile_handles_mutex);
  for (int fd : fds) {
    auto it = g_cufile_handles.find(fd);
    if (it != g_cufile_handles.end()) {
      cuFileHandleDeregister(it->second);
      g_cufile_handles.erase(it);
    }
    close(fd);
  }
}

int64_t register_fd(int raw_fd) {
    CUfileDescr_t descr;
    std::memset(&descr, 0, sizeof(descr));

    // Handle type for standard Linux POSIX raw block devices / files
    descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    descr.handle.fd = raw_fd;

    CUfileHandle_t nvme_handle = nullptr;
    CUfileError_t status = cuFileHandleRegister(&nvme_handle, &descr);

    TORCH_CHECK(
        status.err == CU_FILE_SUCCESS,
        "cuFileHandleRegister failed for fd ", raw_fd,
        " with error code: ", static_cast<int>(status.err)
    );

    // Cast the opaque struct pointer to int64_t for safely passing to Python
    return static_cast<int64_t>(reinterpret_cast<uintptr_t>(nvme_handle));
}


static ThreadPool &getH2DPool() {
  static ThreadPool pool;
  return pool;
}

static ThreadPool &getD2HPool() {
  static ThreadPool pool;
  return pool;
}

void h2d_synchronize() { getH2DPool().synchronize(); }
void d2h_synchronize() { getD2HPool().synchronize(); }

void d2h_copy_async(torch::Tensor src, torch::Tensor dst) {
  AT_ASSERTM(src.is_cuda(), "Source must be a CUDA tensor");
  AT_ASSERTM(!dst.is_cuda(), "Destination must be a CPU tensor");
  AT_ASSERTM(src.is_contiguous(), "Source must be contiguous");
  AT_ASSERTM(dst.is_contiguous(), "Destination must be contiguous");

  auto stream = at::cuda::getCurrentCUDAStream(src.get_device());
  AT_ASSERTM(stream != at::cuda::getDefaultCUDAStream(src.get_device()),
             "Async D2H requires a non-default CUDA stream");

  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half,src.scalar_type(), "d2h_copy_async", [&] {
    getD2HPool().run([=] {
      auto src_ptr = src.data_ptr<scalar_t>();
      auto dst_ptr = dst.data_ptr<scalar_t>();
      cudaMemcpyAsync(dst_ptr, src_ptr, src.numel() * sizeof(scalar_t),
                      cudaMemcpyDeviceToHost, stream);
    });
  });
}

void h2d_copy_async(torch::Tensor src, torch::Tensor dst) {
  AT_ASSERTM(!src.is_cuda(), "Source must be a CPU tensor");
  AT_ASSERTM(dst.is_cuda(), "Destination must be a CUDA tensor");
  AT_ASSERTM(src.is_contiguous(), "Source must be contiguous");
  AT_ASSERTM(dst.is_contiguous(), "Destination must be contiguous");

  auto stream = at::cuda::getCurrentCUDAStream(dst.get_device());
  AT_ASSERTM(stream != at::cuda::getDefaultCUDAStream(dst.get_device()),
             "Async H2D requires a non-default CUDA stream");

  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half,src.scalar_type(), "h2d_copy_async", [&] {
    getH2DPool().run([=] {
      auto src_ptr = src.data_ptr<scalar_t>();
      auto dst_ptr = dst.data_ptr<scalar_t>();
      cudaMemcpyAsync(dst_ptr, src_ptr, src.numel() * sizeof(scalar_t),
                      cudaMemcpyHostToDevice, stream);
    });
  });
}

static void nvme_write(int64_t fd, const void* buf, uint64_t slba, int64_t bytes, int64_t sector_size){
    TORCH_CHECK(bytes > 0 && bytes % sector_size == 0,
                "nvme_write: size ", bytes, " is not a multiple of ", sector_size);

    const int64_t nsec = bytes / sector_size;
    TORCH_CHECK(nsec <= 65536, "nvme_write: ", nsec, " sectors exceeds the 16-bit nblocks limit");

    struct nvme_user_io io;
    std::memset(&io, 0, sizeof(io));
    io.opcode  = 0x01;                                  // nvme_cmd_write
    io.addr    = reinterpret_cast<uint64_t>(buf);
    io.slba    = slba;
    io.nblocks = static_cast<uint16_t>(nsec - 1);       // 0-based

    const int ret = ioctl(static_cast<int>(fd), NVME_IOCTL_SUBMIT_IO, &io);
    TORCH_CHECK(ret == 0,
                "NVMe write failed: slba=", slba, " bytes=", bytes,
                (ret < 0 ? " errno=" : " nvme_status=0x"),
                (ret < 0 ? std::to_string(errno) + " (" + strerror(errno) + ")"
                         : [&]{ char s[8]; snprintf(s, sizeof(s), "%x", ret); return std::string(s); }()));
}

static void nvme_read(int64_t fd, void* buf, uint64_t slba, int64_t bytes, int64_t sector_size){
    TORCH_CHECK(bytes > 0 && bytes % sector_size == 0,
                "nvme_read: size ", bytes, " is not a multiple of ", sector_size);

    const int64_t nsec = bytes / sector_size;
    TORCH_CHECK(nsec <= 65536, "nvme_read: ", nsec, " sectors exceeds the 16-bit nblocks limit");

    struct nvme_user_io io;
    std::memset(&io, 0, sizeof(io));
    io.opcode  = 0x02;                                  // nvme_cmd_read
    io.addr    = reinterpret_cast<uint64_t>(buf);
    io.slba    = slba;
    io.nblocks = static_cast<uint16_t>(nsec - 1);       // 0-based

    const int ret = ioctl(static_cast<int>(fd), NVME_IOCTL_SUBMIT_IO, &io);
    TORCH_CHECK(ret == 0,
                "NVMe read failed: slba=", slba, " bytes=", bytes,
                (ret < 0 ? " errno=" : " nvme_status=0x"),
                (ret < 0 ? std::to_string(errno) + " (" + strerror(errno) + ")"
                         : [&]{ char s[8]; snprintf(s, sizeof(s), "%x", ret); return std::string(s); }()));
}


// Default grinnder
void gather_partitions(int pid, std::vector<torch::Tensor> srcs,
                       torch::Tensor dst,
                       std::vector<torch::Tensor> boundaries) {
  AT_ASSERTM(dst.is_cuda(), "Destination must be a CUDA tensor");
  for (auto &src : srcs) {
    AT_ASSERTM(!src.is_cuda(), "Sources must be CPU tensors");
  }
  AT_ASSERTM(dst.is_contiguous(), "Destination must be contiguous");

  auto stream = at::cuda::getCurrentCUDAStream(dst.get_device());
  AT_ASSERTM(stream != at::cuda::getDefaultCUDAStream(dst.get_device()),
             "Async gather requires a non-default CUDA stream");

  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half,dst.scalar_type(), "gather_partitions", [&] {
    getH2DPool().run([=] {
      auto dst_data = dst.data_ptr<scalar_t>();
      int64_t feat_dim = dst.numel() / dst.size(0);

      // Keep index_select results alive until CUDA copies complete.
      // Without this, selected tensors would be freed when they go
      // out of scope in the loop, but cudaMemcpyAsync DMA may still
      // be reading from their memory (use-after-free).
      std::vector<torch::Tensor> keep_alive;

      // First: copy intra-partition (contiguous block)
      int64_t offset = srcs[pid].size(0);
      auto src_data = srcs[pid].data_ptr<scalar_t>();
      cudaMemcpyAsync(dst_data, src_data,
                      offset * feat_dim * sizeof(scalar_t),
                      cudaMemcpyHostToDevice, stream);

      // Then: copy boundary nodes from each other partition via index_select
      for (size_t i = 0; i < srcs.size(); i++) {
        if ((int)i == pid)
          continue;
        auto bndry = boundaries[i];
        if (bndry.numel() == 0)
          continue;

        torch::Tensor selected = torch::index_select(srcs[i], 0, bndry);
        AT_ASSERTM(!selected.is_cuda(), "Selected must be CPU");
        keep_alive.push_back(selected);

        auto sel_data = selected.data_ptr<scalar_t>();
        int64_t sel_size = selected.size(0);
        cudaMemcpyAsync(dst_data + (offset * feat_dim), sel_data,
                        sel_size * feat_dim * sizeof(scalar_t),
                        cudaMemcpyHostToDevice, stream);
        offset += sel_size;
      }

      AT_ASSERTM(offset == dst.size(0),
                  "Gather: copied size mismatch with destination");

      // Ensure all DMA transfers complete before selected tensors
      // (in keep_alive) are freed. This runs in the worker thread,
      // so main thread stays async.
      cudaStreamSynchronize(stream);
    });
  });
}


#define MAX_BATCH_SIZE 256

// Read tensors directly from partitions
void gather_partitions_direct(
    int pid,
    const std::vector<int>& fds,
    torch::Tensor dst,
    std::vector<torch::Tensor> boundaries) {

  AT_ASSERTM(dst.is_cuda(), "Destination must be a CUDA tensor");
  AT_ASSERTM(dst.is_contiguous(), "Destination must be contiguous");

  auto stream = at::cuda::getCurrentCUDAStream(dst.get_device());
  AT_ASSERTM(stream != at::cuda::getDefaultCUDAStream(dst.get_device()),
             "Async gather requires a non-default CUDA stream");

  getH2DPool().run([=] {
    c10::cuda::CUDAStreamGuard guard(stream);

    int64_t total_rows = dst.size(0);
    int64_t feat_dim = dst.numel() / total_rows;
    int64_t row_bytes = feat_dim * dst.element_size();
    uint8_t* dst_raw = reinterpret_cast<uint8_t*>(dst.data_ptr());

    size_t num_parts = fds.size();
    std::vector<int64_t> part_offsets(num_parts, 0);

    int target_fd = fds[pid];
    CUfileHandle_t target_handle = get_cufile_handle(target_fd);

    off_t target_file_size = lseek(target_fd, 0, SEEK_END);
    int64_t target_nodes = target_file_size / row_bytes;

    int64_t curr_offset = target_nodes;
    size_t total_num_reads = 0;

    for (size_t i = 0; i < num_parts; ++i) {
      if (static_cast<int>(i) != pid && boundaries[i].defined() && boundaries[i].numel() > 0) {
        part_offsets[i] = curr_offset;
        curr_offset += boundaries[i].numel();
        total_num_reads += boundaries[i].numel();
      }
    }

    AT_ASSERTM(curr_offset == total_rows, "Gather offset mismatch with destination size");

    // 1. Direct GDS read for target partition
    // arguments : [ fd, base addr in VRAM, size (to read), file_offset, offset inside VRAM ]
    ssize_t ret = cuFileRead(target_handle, dst_raw, target_file_size, 0, 0);
    AT_ASSERTM(ret == target_file_size, "GDS Read failed for target partition");

    if (total_num_reads == 0) return;

    // -------------------------------------------------------------------
    // 2. Prepare full params list (matching benchmark calloc allocation)
    // -------------------------------------------------------------------
    CUfileIOParams_t *io_params = (CUfileIOParams_t*)calloc(total_num_reads, sizeof(CUfileIOParams_t));
    AT_ASSERTM(io_params != nullptr, "Failed to allocate memory for io_params");

    size_t param_idx = 0;
    for (size_t i = 0; i < num_parts; ++i) {
      if (static_cast<int>(i) == pid || !boundaries[i].defined() || boundaries[i].numel() == 0) {
        continue;
      }

      int fd = fds[i];
      CUfileHandle_t handle = get_cufile_handle(fd);

      auto bndry = boundaries[i].to(at::kCPU).contiguous();
      const int64_t* idx_ptr = bndry.data_ptr<int64_t>();
      int64_t num_boundary_nodes = bndry.numel();
      int64_t start_row_offset = part_offsets[i];

      for (int64_t idx = 0; idx < num_boundary_nodes; ++idx) {
        io_params[param_idx].mode = CUFILE_BATCH;
        io_params[param_idx].opcode = CUFILE_READ;
        io_params[param_idx].fh = handle;
        // See gds_direct_mode() for why the direct path folds the offset into the pointer.
        const int64_t buf_off = (start_row_offset + idx) * row_bytes;
        const bool fold = gds_direct_mode();
        io_params[param_idx].u.batch.devPtr_base = fold ? dst_raw + buf_off : dst_raw;
        io_params[param_idx].u.batch.devPtr_offset = fold ? 0 : buf_off;
        io_params[param_idx].u.batch.file_offset = idx_ptr[idx] * row_bytes;
        io_params[param_idx].u.batch.size = row_bytes;
        param_idx++;
      }
    }

    // -------------------------------------------------------------------
    // 3. Allocate temporary workspace & process in MAX_BATCH_SIZE (4096)
    // -------------------------------------------------------------------
    CUfileBatchHandle_t batch_handle;
    CUfileError_t status = cuFileBatchIOSetUp(&batch_handle, MAX_BATCH_SIZE);
    AT_ASSERTM(status.err == CU_FILE_SUCCESS, "cuFileBatchIOSetUp failed");

    CUfileIOEvents_t events[MAX_BATCH_SIZE];

    for (size_t offset_idx = 0; offset_idx < total_num_reads; offset_idx += MAX_BATCH_SIZE) {
      unsigned int current_batch_size = (total_num_reads - offset_idx > MAX_BATCH_SIZE)
                                        ? MAX_BATCH_SIZE
                                        : (unsigned int)(total_num_reads - offset_idx);

      status = cuFileBatchIOSubmit(batch_handle, current_batch_size, &io_params[offset_idx], 0);
      AT_ASSERTM(status.err == CU_FILE_SUCCESS, "cuFileBatchIOSubmit failed");

      unsigned int completed = current_batch_size;
      status = cuFileBatchIOGetStatus(batch_handle, current_batch_size, &completed, events, NULL);
      AT_ASSERTM(status.err == CU_FILE_SUCCESS, "cuFileBatchIOGetStatus failed");
    }

    // Cleanup batch resources
    cuFileBatchIODestroy(batch_handle);
    free(io_params);
  });
}


#define PAGE_SIZE 16*1024


static inline uint64_t encode_row_request_ppa(
    int rows_per_page,
    uint64_t raw_ppa,
    int64_t row_offset,
    bool feature_type)
{
    constexpr uint64_t PL_SHIFT = 40;
    constexpr uint64_t PL_MASK = 0xFULL;

    constexpr uint64_t TYPE_SHIFT = 62;
    constexpr uint64_t TYPE_BIT = 1ULL << TYPE_SHIFT;


    TORCH_CHECK(
        row_offset >= 0 && row_offset < rows_per_page,
        "Invalid row offset: ", row_offset
    );

    // Mapped PPAs must use plane 0. The request uses
    // the plane field to carry the row index instead.
    TORCH_CHECK(
        ((raw_ppa >> PL_SHIFT) & PL_MASK) == 0,
        "Expected mapped PPA with plane = 0"
    );

    uint64_t request = (raw_ppa & ~(PL_MASK << PL_SHIFT)) | (static_cast<uint64_t>(row_offset) << PL_SHIFT);
    

    // set or clear the custom type bit
    if (feature_type) {
        request |= TYPE_BIT;
    } else {
        request &= ~TYPE_BIT;
    }

    return request;
}

static inline uint64_t encode_host_ftl_bit(uint64_t raw_ppa)
{
    constexpr uint64_t HOST_FTL_BIT = 1ULL << 61;
    return raw_ppa |= HOST_FTL_BIT;
}

// gather_partitions_direct_raw / gather_partitions_direct_raw_async are
// defined below gpu_write: they share its cuFile batch runner.


void gather_activations_direct(
    const std::string& filepath,
    torch::Tensor dst) {

  AT_ASSERTM(dst.is_cuda(), "Destination must be a CUDA tensor");
  AT_ASSERTM(dst.is_contiguous(), "Destination must be contiguous");

  auto stream = at::cuda::getCurrentCUDAStream(dst.get_device());
  AT_ASSERTM(stream != at::cuda::getDefaultCUDAStream(dst.get_device()),
             "Async gather requires a non-default CUDA stream");

  getH2DPool().run([=] {
    c10::cuda::CUDAStreamGuard guard(stream);

    int fd = open(filepath.c_str(), O_RDONLY | O_DIRECT);
    AT_ASSERTM(fd >= 0, "Failed to open file: " + filepath);

    off_t file_size = lseek(fd, 0, SEEK_END);
    AT_ASSERTM(file_size > 0, "Failed to get valid file size via lseek");
    AT_ASSERTM(dst.nbytes() >= static_cast<size_t>(file_size),
               "Destination tensor is smaller than the file size");

    CUfileHandle_t handle;
    CUfileDescr_t descr;
    memset(&descr, 0, sizeof(CUfileDescr_t));
    descr.handle.fd = fd;
    descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    CUfileError_t status = cuFileHandleRegister(&handle, &descr);
    if (status.err != CU_FILE_SUCCESS) {
      close(fd);
      AT_ERROR("cuFileHandleRegister failed for file: " + filepath);
    }

    uint8_t* dst_raw = reinterpret_cast<uint8_t*>(dst.data_ptr());

    // 3. Direct GDS read for the full file
    ssize_t ret = cuFileRead(handle, dst_raw, file_size, 0, 0);

    // 4. Clean up GDS handle and file descriptor
    cuFileHandleDeregister(handle);
    close(fd);

    AT_ASSERTM(ret == file_size, "GDS Read failed for full file (activations)");
  });
}


void device_write(
    int64_t fd,
    torch::Tensor src,
    torch::Tensor runs,
    int64_t sector_size,
    int64_t page_size
) {
    TORCH_CHECK(!src.is_cuda() && src.is_contiguous(), "src must be contiguous CPU");
    TORCH_CHECK(runs.dim() == 2 && runs.size(1) == 3 && runs.scalar_type() == at::kLong);

    auto r = runs.contiguous();
    const int64_t* rp = r.data_ptr<int64_t>();
    const uint8_t* src_raw = static_cast<const uint8_t*>(src.data_ptr());
    const int64_t total_bytes = src.numel() * src.element_size();
    const int64_t MAX_IO = 128 * 1024;    
    std::vector<uint8_t> tail(page_size); 

    for(int64_t k = 0; k < r.size(0); ++k){
        const int64_t file_off = rp[3*k], buf_off = rp[3*k+1], size = rp[3*k+2];

        for(int64_t done = 0; done < size; done += MAX_IO){
            const int64_t chunk = std::min(MAX_IO, size - done);
            const int64_t b = buf_off + done;
            const uint8_t* data = src_raw + b;
            int64_t n = chunk;

            if(b + chunk > total_bytes){
                const int64_t full = (total_bytes - b) / page_size * page_size;
                if (full > 0) {                      // write the full pages first
                    nvme_write(fd, data, (file_off + done) / sector_size, full, sector_size);
                }

                std::fill(tail.begin(), tail.end(), 0);
                std::memcpy(tail.data(), data + full, total_bytes - b - full);
                nvme_write(fd, tail.data(), (file_off + done + full) / sector_size,
                           page_size, sector_size);
                continue;
            }
            nvme_write(fd, data, (file_off + done) / sector_size, n, sector_size);
        }
    } 
}



// ---------------------------------------------------------------------------
// GPU -> SSD writes with the cuFile batch API, run asynchronously.
//
// gpu_write_async() returns immediately with a job id. A worker thread waits
// for the kernels that produced `src` (a CUDA event recorded on the caller's
// current stream), then submits the writes in batches of up to
// GPU_WRITE_BATCH entries and waits for each batch to complete.
// gpu_write_wait(id) / gpu_write_wait_all() block until done and rethrow errors.
//
// The job holds a reference to `src`, but the caller must still not free
// (resize_(0)) or overwrite it before waiting.
// ---------------------------------------------------------------------------

// Must not exceed "io_batchsize" in cufile.json (default 128).
static constexpr unsigned GPU_WRITE_BATCH = 128;
static constexpr int64_t GPU_WRITE_MAX_IO = 128 * 1024;

struct GpuWriteIo {
    void* base;        // devPtr_base
    int64_t buf_off;   // devPtr_offset
    int64_t dev_off;   // file_offset (dev_lba * sector_size)
    int64_t size;
};

static std::mutex g_gpu_write_mutex;
// Serializes cuFile batches when gds_direct_mode() is on.
static std::mutex g_cufile_batch_mutex;
static std::unordered_map<int64_t, std::future<void>> g_gpu_write_jobs;
static int64_t g_gpu_write_next_id = 0;

// Runs every I/O in `ios` (CUFILE_READ or CUFILE_WRITE) through the cuFile
// batch API, GPU_WRITE_BATCH entries at a time. Shared by GPU writes and gathers.
static void gpu_write_run_batches(CUfileHandle_t handle, const std::vector<GpuWriteIo>& ios,
                                  CUfileOpcode_t opcode) {
    if (ios.empty()) return;
    const char* what = opcode == CUFILE_READ ? "read" : "write";
    std::unique_lock<std::mutex> batch_lock(g_cufile_batch_mutex, std::defer_lock);
    if (gds_direct_mode()) batch_lock.lock();

    CUfileBatchHandle_t batch;
    CUfileError_t st = cuFileBatchIOSetUp(&batch, GPU_WRITE_BATCH);
    TORCH_CHECK(st.err == CU_FILE_SUCCESS, "cuFileBatchIOSetUp failed: err=", static_cast<int>(st.err));

    std::vector<CUfileIOParams_t> params(GPU_WRITE_BATCH);
    std::vector<CUfileIOEvents_t> events(GPU_WRITE_BATCH);
    std::string error;

    for (size_t first = 0; first < ios.size() && error.empty(); first += GPU_WRITE_BATCH) {
        const unsigned n = static_cast<unsigned>(std::min<size_t>(GPU_WRITE_BATCH, ios.size() - first));

        for (unsigned i = 0; i < n; ++i) {
            const GpuWriteIo& io = ios[first + i];
            std::memset(&params[i], 0, sizeof(CUfileIOParams_t));
            params[i].mode = CUFILE_BATCH;
            params[i].fh = handle;
            params[i].opcode = opcode;
            params[i].cookie = reinterpret_cast<void*>(static_cast<uintptr_t>(first + i));
            // See gds_direct_mode() for why the direct path folds the offset into the pointer.
            const bool fold = gds_direct_mode();
            params[i].u.batch.devPtr_base = fold ? static_cast<char*>(io.base) + io.buf_off : io.base;
            params[i].u.batch.devPtr_offset = fold ? 0 : io.buf_off;
            params[i].u.batch.file_offset = io.dev_off;
            params[i].u.batch.size = static_cast<size_t>(io.size);
        }

        st = cuFileBatchIOSubmit(batch, n, params.data(), 0);
        if (st.err != CU_FILE_SUCCESS) {
            error = "cuFileBatchIOSubmit failed: err=" + std::to_string(static_cast<int>(st.err));
            break;
        }

        // Wait for every entry of this batch before reusing the batch handle.
        unsigned completed = 0;
        while (completed < n) {
            unsigned nr = n - completed;
            struct timespec timeout = {1, 0};
            st = cuFileBatchIOGetStatus(batch, 1, &nr, events.data(), &timeout);
            if (st.err != CU_FILE_SUCCESS) {
                error = "cuFileBatchIOGetStatus failed: err=" + std::to_string(static_cast<int>(st.err));
                break;
            }
            for (unsigned i = 0; i < nr; ++i) {
                const size_t idx = reinterpret_cast<uintptr_t>(events[i].cookie);
                const GpuWriteIo& io = ios[idx];
                if ((events[i].status != CUFILE_COMPLETE ||
                     events[i].ret != static_cast<size_t>(io.size)) && error.empty()) {
                    error = std::string("cuFile batch ") + what + " failed: dev_off=" + std::to_string(io.dev_off) +
                            " lba=" + std::to_string(io.dev_off / 512) +
                            " buf_off=" + std::to_string(io.buf_off) +
                            " bytes=" + std::to_string(io.size) +
                            " status=" + std::to_string(static_cast<int>(events[i].status)) +
                            " ret=" + std::to_string(events[i].ret);
                }
            }
            completed += nr;
        }
    }

    cuFileBatchIODestroy(batch);
    TORCH_CHECK(error.empty(), error);
}

int64_t gpu_write_async(
    int64_t dev_handle_ptr,      /* cuFile handle from register_fd */
    torch::Tensor src,           /* contiguous CUDA tensor */
    torch::Tensor runs,          /* [N, 3] int64: (dev_off, buf_off, size), same as device_write */
    int64_t sector_size,
    int64_t page_size
) {
    TORCH_CHECK(src.is_cuda() && src.is_contiguous(), "src must be a contiguous CUDA tensor");
    TORCH_CHECK(runs.dim() == 2 && runs.size(1) == 3 && runs.scalar_type() == at::kLong,
                "runs must be an [N, 3] int64 tensor");
    TORCH_CHECK(page_size > 0 && page_size % sector_size == 0,
                "page_size must be a multiple of sector_size");

    CUfileHandle_t handle =
        reinterpret_cast<CUfileHandle_t>(static_cast<uintptr_t>(dev_handle_ptr));
    const int device = src.get_device();

    // Mark the point on the caller's stream where src is fully produced.
    cudaEvent_t ready;
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    C10_CUDA_CHECK(cudaEventRecord(ready, at::cuda::getCurrentCUDAStream(device)));

    auto r = runs.to(at::kCPU).contiguous();

    auto job = [handle, device, ready, src, r, sector_size, page_size]() {
        c10::cuda::CUDAGuard guard(device);
        C10_CUDA_CHECK(cudaEventSynchronize(ready));
        C10_CUDA_CHECK(cudaEventDestroy(ready));

        const int64_t* rp = r.data_ptr<int64_t>();
        void* src_raw = src.data_ptr();
        const int64_t total_bytes = src.numel() * src.element_size();
        torch::Tensor tail;          // page-sized GPU scratch for the last partial page
        std::vector<GpuWriteIo> ios;

        // Same run/chunk/tail layout as device_write, so reads of this
        // tensor (gather, device_read_and_verify) line up exactly.
        for (int64_t k = 0; k < r.size(0); ++k) {
            const int64_t file_off = rp[3*k], buf_off = rp[3*k+1], size = rp[3*k+2];

            for (int64_t done = 0; done < size; done += GPU_WRITE_MAX_IO) {
                const int64_t chunk = std::min(GPU_WRITE_MAX_IO, size - done);
                const int64_t b = buf_off + done;
                if (b >= total_bytes) break;

                if (b + chunk > total_bytes) {
                    const int64_t full = (total_bytes - b) / page_size * page_size;
                    if (full > 0) {
                        ios.push_back({src_raw, b, file_off + done, full});
                    }
                    // Last partial page: copy into a zeroed page buffer, write the whole page
                    // zero-fill and copy on the same stream so they stay ordered
                    auto stream = c10::cuda::getStreamFromPool(false, device);
                    c10::cuda::CUDAStreamGuard stream_guard(stream);
                    tail = torch::zeros({page_size}, src.options().dtype(at::kByte));
                    C10_CUDA_CHECK(cudaMemcpyAsync(
                        tail.data_ptr(),
                        static_cast<const uint8_t*>(src_raw) + b + full,
                        total_bytes - b - full,
                        cudaMemcpyDeviceToDevice,
                        stream));
                    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
                    ios.push_back({tail.data_ptr(), 0, file_off + done + full, page_size});
                    continue;
                }
                ios.push_back({src_raw, b, file_off + done, chunk});
            }
        }

        for (const GpuWriteIo& io : ios) {
            TORCH_CHECK(io.dev_off % sector_size == 0 && io.size % sector_size == 0,
                        "gpu_write: unaligned I/O dev_off=", io.dev_off, " size=", io.size);
        }

        gpu_write_run_batches(handle, ios, CUFILE_WRITE);   // `tail` stays alive until this returns
    };

    std::lock_guard<std::mutex> lock(g_gpu_write_mutex);
    const int64_t id = g_gpu_write_next_id++;
    g_gpu_write_jobs.emplace(id, std::async(std::launch::async, std::move(job)));
    return id;
}

void gpu_write_wait(int64_t id) {
    std::future<void> fut;
    {
        std::lock_guard<std::mutex> lock(g_gpu_write_mutex);
        auto it = g_gpu_write_jobs.find(id);
        if (it == g_gpu_write_jobs.end()) return;   // already waited
        fut = std::move(it->second);
        g_gpu_write_jobs.erase(it);
    }
    fut.get();   // rethrows the worker's error, if any
}

void gpu_write_wait_all() {
    std::vector<int64_t> ids;
    {
        std::lock_guard<std::mutex> lock(g_gpu_write_mutex);
        for (auto& kv : g_gpu_write_jobs) ids.push_back(kv.first);
    }
    std::sort(ids.begin(), ids.end());
    std::string first_error;
    for (int64_t id : ids) {
        try {
            gpu_write_wait(id);
        } catch (const std::exception& e) {
            if (first_error.empty()) first_error = e.what();
        }
    }
    TORCH_CHECK(first_error.empty(), first_error);
}

// Blocking version: submit and wait.
void gpu_write(
    int64_t dev_handle_ptr,
    torch::Tensor src,
    torch::Tensor runs,
    int64_t sector_size,
    int64_t page_size
) {
    gpu_write_wait(gpu_write_async(dev_handle_ptr, src, runs, sector_size, page_size));
}


// ---------------------------------------------------------------------------
// SSD -> GPU gather with the cuFile batch API, run asynchronously.
//
// gather_partitions_direct_raw_async() returns a job id at once. A worker
// thread waits until `dst` is free to overwrite (a CUDA event recorded on the
// caller's current stream), then reads every run into `dst` in batched
// chunks. The data is in `dst` only after gather_wait(id) returns; no CUDA
// stream knows about these reads, so wait before any kernel uses `dst`.
// ---------------------------------------------------------------------------

int64_t gather_partitions_direct_raw_async(
    int64_t dev_handle_ptr,
    torch::Tensor dst,
    torch::Tensor runs,          /* [N, 3] int64: (dev_off, dst_off, size) in bytes */
    int64_t sector_size
) {
    TORCH_CHECK(dst.is_cuda() && dst.is_contiguous(), "dst must be a contiguous CUDA tensor");
    TORCH_CHECK(runs.dim() == 2 && runs.size(1) == 3 && runs.scalar_type() == at::kLong,
                "runs must be an [N, 3] int64 tensor");

    CUfileHandle_t handle =
        reinterpret_cast<CUfileHandle_t>(static_cast<uintptr_t>(dev_handle_ptr));
    const int device = dst.get_device();
    const int64_t dst_bytes = dst.numel() * dst.element_size();

    // Build the chunk list now, so bad runs fail in the caller.
    auto r = runs.to(at::kCPU).contiguous();
    const int64_t* rp = r.data_ptr<int64_t>();
    void* dst_raw = dst.data_ptr();
    std::vector<GpuWriteIo> ios;
    for (int64_t k = 0; k < r.size(0); ++k) {
        const int64_t dev_off = rp[3*k], buf_off = rp[3*k+1], size = rp[3*k+2];
        TORCH_CHECK(dev_off % sector_size == 0 && size % sector_size == 0,
                    "gather run ", k, " not sector-aligned: dev_off=", dev_off, " size=", size);
        TORCH_CHECK(buf_off >= 0 && buf_off + size <= dst_bytes,
                    "gather run ", k, " exceeds dst: buf_off=", buf_off,
                    " size=", size, " dst_bytes=", dst_bytes);
        for (int64_t done = 0; done < size; done += GPU_WRITE_MAX_IO) {
            ios.push_back({dst_raw, buf_off + done, dev_off + done,
                           std::min(GPU_WRITE_MAX_IO, size - done)});
        }
    }

    // Kernels still using this memory (e.g. a freed tensor it was reused from)
    // must finish before the reads overwrite it.
    cudaEvent_t ready;
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    C10_CUDA_CHECK(cudaEventRecord(ready, at::cuda::getCurrentCUDAStream(device)));

    auto job = [handle, device, ready, dst, ios = std::move(ios)]() {
        c10::cuda::CUDAGuard guard(device);
        C10_CUDA_CHECK(cudaEventSynchronize(ready));
        C10_CUDA_CHECK(cudaEventDestroy(ready));
        gpu_write_run_batches(handle, ios, CUFILE_READ);   // `dst` held alive until done
    };

    std::lock_guard<std::mutex> lock(g_gpu_write_mutex);
    const int64_t id = g_gpu_write_next_id++;
    g_gpu_write_jobs.emplace(id, std::async(std::launch::async, std::move(job)));
    return id;
}

// ---------------------------------------------------------------------------
// GPU <-> SSD transfers without cuFile.
//
// gather_partitions_nvme_async / gpu_write_nvme_async use the same runs,
// chunking and job/wait API as the cuFile versions, but move data with NVMe
// passthrough ioctls (nvme_read / nvme_write) on pinned host memory plus
// cudaMemcpyAsync. I/Os are processed in windows of NVME_WINDOW bytes. Each
// window's ioctls run on GRINNDER_NVME_THREADS threads (default 16), because
// every ioctl blocks until its command completes.
// ---------------------------------------------------------------------------

static constexpr int64_t NVME_WINDOW = 64LL << 20;

static int nvme_io_threads() {
    static const int n = [] {
        const char* v = std::getenv("GRINNDER_NVME_THREADS");
        const int x = v ? std::atoi(v) : 0;
        return x > 0 ? x : 16;
    }();
    return n;
}

// Runs fn(0..n-1) on up to nvme_io_threads() threads. Stops handing out work
// after the first exception and rethrows its message once all threads joined.
static void run_nvme_parallel(size_t n, const std::function<void(size_t)>& fn) {
    std::atomic<size_t> cursor{0};
    std::atomic<bool> failed{false};
    std::string error;
    std::mutex error_mutex;
    auto worker = [&] {
        for (size_t i; !failed && (i = cursor.fetch_add(1)) < n;) {
            try {
                fn(i);
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!failed.exchange(true)) error = e.what();
            }
        }
    };
    std::vector<std::thread> threads;
    const size_t nt = std::min<size_t>(nvme_io_threads(), n);
    for (size_t t = 0; t < nt; ++t) threads.emplace_back(worker);
    for (auto& t : threads) t.join();
    TORCH_CHECK(error.empty(), error);
}

// ---------------------------------------------------------------------------
// Vector reads (FEMU blackbox SSD built with HOST_FTL, opcode 0x92).
//
// One command carries a list of up to N physical 512 B sector addresses and
// returns their data packed in list order. Buffer layout of one command:
//   [ u64 addr[nr] | padding to 4 KiB ][ nr * 512 B data ]
// cdw12[15:0] = nr - 1, cdw13 = 0 (list in the data buffer's first page).
// The buffer must be page aligned, or the kernel bounces it and the device
// would not see the list.
//
// Small I/Os (boundary rows) are packed into vector commands; large ones
// (own-partition runs, whole tensors) keep using nvme_read.
//   GRINNDER_NVME_VECTOR=1          turn it on
//   GRINNDER_NVME_VECTOR_MAX=64     sectors per command (max 512)
//   GRINNDER_NVME_VECTOR_MIN_IO=32768  pack I/Os smaller than this (bytes)
// If the device or kernel refuses the command, vector reads are switched
// off for the rest of the run and the reads are redone one I/O at a time.
// ---------------------------------------------------------------------------

static constexpr int64_t VEC_SECTOR = 512;
static constexpr int64_t VEC_LIST_BYTES = 4096;
static constexpr int VEC_HARD_MAX = 512;            // FEMU BB_MAX_VEC
static constexpr uint8_t NVME_OPC_VECTOR_READ = 0x92;

struct NvmeVecConfig {
    bool on;
    int max_entries;
    int64_t min_io;
};

static const NvmeVecConfig& nvme_vec_config() {
    static const NvmeVecConfig cfg = [] {
        NvmeVecConfig c{false, 64, 32 * 1024};
        if (const char* v = std::getenv("GRINNDER_NVME_VECTOR")) c.on = std::atoi(v) != 0;
        if (const char* v = std::getenv("GRINNDER_NVME_VECTOR_MAX")) {
            const int x = std::atoi(v);
            if (x > 0) c.max_entries = std::min(x, VEC_HARD_MAX);
        }
        if (const char* v = std::getenv("GRINNDER_NVME_VECTOR_MIN_IO")) {
            const long long x = std::atoll(v);
            if (x > 0) c.min_io = x;
        }
        if (c.on) {
            std::fprintf(stderr,
                         "[nvme] vector reads on: up to %d sectors per command, "
                         "packing I/Os smaller than %lld B\n",
                         c.max_entries, static_cast<long long>(c.min_io));
        }
        return c;
    }();
    return cfg;
}

// Set once a vector command is refused; from then on plain reads only.
static std::atomic<bool> g_vec_disabled{false};

static void disable_vector_reads(const std::string& why) {
    if (!g_vec_disabled.exchange(true)) {
        std::fprintf(stderr, "[nvme] vector reads off: %s. Using one read per I/O.\n",
                     why.c_str());
    }
}

// Namespace id of the block device behind fd (cached per fd); 0 if unknown.
static uint32_t nvme_nsid(int64_t fd) {
    static std::mutex m;
    static std::unordered_map<int64_t, uint32_t> cache;
    std::lock_guard<std::mutex> lock(m);
    auto it = cache.find(fd);
    if (it != cache.end()) return it->second;
    const int r = ioctl(static_cast<int>(fd), NVME_IOCTL_ID);
    const uint32_t nsid = r > 0 ? static_cast<uint32_t>(r) : 0;
    cache.emplace(fd, nsid);
    return nsid;
}

// One vector read. region: page-aligned, list already in its first nr*8 bytes,
// data lands at region + VEC_LIST_BYTES. Returns 0, an NVMe status (> 0) or -errno.
static int nvme_vread(int64_t fd, uint32_t nsid, uint8_t* region, int nr) {
    struct nvme_passthru_cmd c;
    std::memset(&c, 0, sizeof(c));
    c.opcode = NVME_OPC_VECTOR_READ;
    c.nsid = nsid;
    c.addr = reinterpret_cast<uint64_t>(region);
    c.data_len = static_cast<uint32_t>(VEC_LIST_BYTES + nr * VEC_SECTOR);
    c.cdw12 = static_cast<uint32_t>(nr - 1);
    c.cdw13 = 0;
    const int ret = ioctl(static_cast<int>(fd), NVME_IOCTL_IO_CMD, &c);
    return ret < 0 ? -errno : ret;
}

static std::string nvme_ret_str(int ret) {
    char s[64];
    if (ret < 0) std::snprintf(s, sizeof(s), "errno=%d (%s)", -ret, strerror(-ret));
    else std::snprintf(s, sizeof(s), "nvme_status=0x%x", ret);
    return s;
}

// Errors that mean "this device/kernel can't do vector reads" (fall back),
// as opposed to a failed read (report it).
//   Invalid Command Opcode: FEMU built without HOST_FTL.
//   Invalid Field: FEMU refused the command itself (list + data > MDTS,
//     LBA format size != FTL sector size, buffer in the CMB, ...).
//   EINVAL/EPERM/EACCES/ENOTTY: the kernel refused it (size above
//     max_hw_sectors, no CAP_SYS_ADMIN, not an NVMe device).
// An LBA Out of Range (0x80) or any other status is a real read error.
static bool vector_unsupported(int ret) {
    if (ret > 0) {
        const int sc = ret & 0x7ff;                   // SCT (10:8) | SC (7:0)
        return sc == 0x001 || sc == 0x002;
    }
    return ret == -EINVAL || ret == -EPERM || ret == -EACCES || ret == -ENOTTY;
}

int64_t gather_partitions_nvme_async(
    int64_t fd,
    torch::Tensor dst,
    torch::Tensor runs,          /* [N, 3] int64: (dev_off, dst_off, size) in bytes */
    int64_t sector_size
) {
    TORCH_CHECK(dst.is_cuda() && dst.is_contiguous(), "dst must be a contiguous CUDA tensor");
    TORCH_CHECK(runs.dim() == 2 && runs.size(1) == 3 && runs.scalar_type() == at::kLong,
                "runs must be an [N, 3] int64 tensor");

    const int device = dst.get_device();
    const int64_t dst_bytes = dst.numel() * dst.element_size();

    // Build the chunk list now, so bad runs fail in the caller.
    // A run may extend past the end of dst (the zero-padded last page of a
    // whole-tensor read): the full sectors are read, but only the bytes that
    // fall inside dst are copied (`copy` below).
    struct ReadIo {
        int64_t buf_off;   // first byte in dst
        int64_t dev_off;   // device byte offset
        int64_t size;      // bytes read from the device (sector multiple)
        int64_t copy;      // bytes copied to dst (<= size)
    };
    auto r = runs.to(at::kCPU).contiguous();
    const int64_t* rp = r.data_ptr<int64_t>();
    std::vector<ReadIo> ios;
    for (int64_t k = 0; k < r.size(0); ++k) {
        const int64_t dev_off = rp[3*k], buf_off = rp[3*k+1], size = rp[3*k+2];
        TORCH_CHECK(dev_off % sector_size == 0 && size % sector_size == 0,
                    "gather run ", k, " not sector-aligned: dev_off=", dev_off, " size=", size);
        TORCH_CHECK(buf_off >= 0 && buf_off < dst_bytes,
                    "gather run ", k, " starts outside dst: buf_off=", buf_off,
                    " dst_bytes=", dst_bytes);
        for (int64_t done = 0; done < size; done += GPU_WRITE_MAX_IO) {
            const int64_t b = buf_off + done;
            if (b >= dst_bytes) break;   // rest of the run is padding
            const int64_t chunk = std::min(GPU_WRITE_MAX_IO, size - done);
            ios.push_back({b, dev_off + done, chunk, std::min(chunk, dst_bytes - b)});
        }
    }

    // Kernels still using this memory must finish before the copies overwrite it.
    cudaEvent_t ready;
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    C10_CUDA_CHECK(cudaEventRecord(ready, at::cuda::getCurrentCUDAStream(device)));

    auto job = [fd, device, ready, dst, sector_size, ios = std::move(ios)]() {
        c10::cuda::CUDAGuard guard(device);
        C10_CUDA_CHECK(cudaEventSynchronize(ready));
        C10_CUDA_CHECK(cudaEventDestroy(ready));
        if (ios.empty()) return;

        auto stream = c10::cuda::getStreamFromPool(false, device);
        const auto pinned = torch::TensorOptions().dtype(torch::kUInt8).pinned_memory(true);
        torch::Tensor stage[2] = {torch::empty({NVME_WINDOW}, pinned),
                                  torch::empty({NVME_WINDOW}, pinned)};
        cudaEvent_t copied[2];
        bool pending[2] = {false, false};
        for (auto& e : copied) C10_CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));

        uint8_t* dst_raw = static_cast<uint8_t*>(dst.data_ptr());
        std::string error;

        // Vector reads: only for 512 B sectors (the list holds sector addresses).
        const NvmeVecConfig& vcfg = nvme_vec_config();
        const bool vec = vcfg.on && !g_vec_disabled.load() && sector_size == VEC_SECTOR;
        const uint32_t nsid = vec ? nvme_nsid(fd) : 0;
        if (vec && nsid == 0) disable_vector_reads("NVME_IOCTL_ID failed on the device fd");
        const int64_t vec_max_bytes = static_cast<int64_t>(vcfg.max_entries) * VEC_SECTOR;
        auto packable = [&](const ReadIo& io) {
            return io.size < vcfg.min_io && io.size <= vec_max_bytes;
        };

        // One command: a single plain read, or a vector read of `count`
        // consecutive I/Os. `region` is its staging offset (vector: the list
        // page; data follows at region + VEC_LIST_BYTES).
        struct Cmd {
            size_t first;     // index of its first I/O within the window
            size_t count;
            bool vector;
            int64_t region;
        };

        size_t next = 0;
        int buf = 0;
        while (next < ios.size() && error.empty()) {
            // The GPU copy out of this buffer (two windows ago) must be done first.
            if (pending[buf]) {
                C10_CUDA_CHECK(cudaEventSynchronize(copied[buf]));
                pending[buf] = false;
            }
            uint8_t* st = stage[buf].data_ptr<uint8_t>();
            const bool use_vec = vec && !g_vec_disabled.load();
            // Vector regions start on a page boundary of the actual address.
            const int64_t base_pad =
                (-static_cast<int64_t>(reinterpret_cast<uintptr_t>(st))) & (VEC_LIST_BYTES - 1);

            // Next window: as many commands as fit in one staging buffer.
            // Plain I/Os sit back to back; each vector command takes a page
            // aligned [list page | data] region.
            const size_t first = next;
            std::vector<int64_t> stage_off;   // per I/O of the window
            std::vector<Cmd> cmds;
            int64_t used = 0;
            while (next < ios.size()) {
                if (use_vec && packable(ios[next])) {
                    const int64_t start =
                        base_pad + (((used - base_pad) + VEC_LIST_BYTES - 1) & ~(VEC_LIST_BYTES - 1));
                    size_t j = next;
                    int64_t data = 0;
                    while (j < ios.size() && packable(ios[j]) &&
                           data + ios[j].size <= vec_max_bytes) {
                        data += ios[j].size;
                        ++j;
                    }
                    if (start + VEC_LIST_BYTES + data > NVME_WINDOW) break;
                    int64_t off = start + VEC_LIST_BYTES;
                    for (size_t k = next; k < j; ++k) {
                        stage_off.push_back(off);
                        off += ios[k].size;
                    }
                    cmds.push_back({next - first, j - next, true, start});
                    used = off;
                    next = j;
                } else {
                    if (used + ios[next].size > NVME_WINDOW) break;
                    stage_off.push_back(used);
                    cmds.push_back({next - first, 1, false, used});
                    used += ios[next].size;
                    ++next;
                }
            }
            const size_t n = next - first;

            try {
                run_nvme_parallel(cmds.size(), [&](size_t c) {
                    const Cmd& cmd = cmds[c];
                    if (cmd.vector && !g_vec_disabled.load()) {
                        uint8_t* region = st + cmd.region;
                        uint64_t* list = reinterpret_cast<uint64_t*>(region);
                        int e = 0;
                        for (size_t k = cmd.first; k < cmd.first + cmd.count; ++k) {
                            const ReadIo& io = ios[first + k];
                            const uint64_t lba = static_cast<uint64_t>(io.dev_off / VEC_SECTOR);
                            for (int64_t s = 0; s < io.size / VEC_SECTOR; ++s) list[e++] = lba + s;
                        }
                        const int ret = nvme_vread(fd, nsid, region, e);
                        if (ret == 0) return;
                        TORCH_CHECK(vector_unsupported(ret),
                                    "NVMe vector read failed: ", e, " sectors, first lba=",
                                    list[0], " ", nvme_ret_str(ret));
                        disable_vector_reads("vector read refused (" + nvme_ret_str(ret) +
                                             "); if errno=22, try a smaller "
                                             "GRINNDER_NVME_VECTOR_MAX");
                    }
                    // Plain reads (also the fallback for a refused vector command).
                    for (size_t k = cmd.first; k < cmd.first + cmd.count; ++k) {
                        const ReadIo& io = ios[first + k];
                        nvme_read(fd, st + stage_off[k], io.dev_off / sector_size, io.size,
                                  sector_size);
                    }
                });
            } catch (const std::exception& e) {
                error = e.what();    // still wait for copies already queued below
                break;
            }

            // One copy per span that is contiguous in dst and in staging
            // (usually long runs; vector regions end at their list pages).
            // A clipped I/O (copy < size) ends a span: its padding is not copied.
            for (size_t i = 0; i < n;) {
                int64_t len = ios[first + i].copy;
                size_t j = i + 1;
                while (j < n && ios[first + j - 1].copy == ios[first + j - 1].size &&
                       ios[first + j].buf_off == ios[first + i].buf_off + len &&
                       stage_off[j] == stage_off[i] + len) {
                    len += ios[first + j].copy;
                    ++j;
                }
                C10_CUDA_CHECK(cudaMemcpyAsync(dst_raw + ios[first + i].buf_off, st + stage_off[i],
                                               len, cudaMemcpyHostToDevice, stream));
                i = j;
            }
            C10_CUDA_CHECK(cudaEventRecord(copied[buf], stream));
            pending[buf] = true;
            buf ^= 1;
        }

        // `dst` holds the data once gather_wait() returns.
        C10_CUDA_CHECK(cudaStreamSynchronize(stream));
        for (auto& e : copied) cudaEventDestroy(e);
        TORCH_CHECK(error.empty(), error);
    };

    std::lock_guard<std::mutex> lock(g_gpu_write_mutex);
    const int64_t id = g_gpu_write_next_id++;
    g_gpu_write_jobs.emplace(id, std::async(std::launch::async, std::move(job)));
    return id;
}

// GPU -> SSD write without cuFile: copies `src` to pinned host memory one
// window at a time, then writes it with nvme_write. Same runs and tail-page
// layout as device_write / gpu_write_async, so the existing reads line up.
// Finish with gpu_write_wait(id).
int64_t gpu_write_nvme_async(
    int64_t fd,
    torch::Tensor src,           /* contiguous CUDA tensor */
    torch::Tensor runs,          /* [N, 3] int64: (dev_off, buf_off, size), same as device_write */
    int64_t sector_size,
    int64_t page_size
) {
    TORCH_CHECK(src.is_cuda() && src.is_contiguous(), "src must be a contiguous CUDA tensor");
    TORCH_CHECK(runs.dim() == 2 && runs.size(1) == 3 && runs.scalar_type() == at::kLong,
                "runs must be an [N, 3] int64 tensor");
    TORCH_CHECK(page_size > 0 && page_size % sector_size == 0 && page_size <= NVME_WINDOW,
                "page_size must be a multiple of sector_size");

    struct WriteIo {
        int64_t buf_off;   // first byte in src
        int64_t dev_off;   // device byte offset
        int64_t size;      // bytes written to the device
        int64_t valid;     // bytes taken from src; the rest (tail page) is zeros
    };

    const int device = src.get_device();
    const int64_t total_bytes = src.numel() * src.element_size();
    auto r = runs.to(at::kCPU).contiguous();
    const int64_t* rp = r.data_ptr<int64_t>();

    std::vector<WriteIo> ios;
    for (int64_t k = 0; k < r.size(0); ++k) {
        const int64_t file_off = rp[3*k], buf_off = rp[3*k+1], size = rp[3*k+2];
        for (int64_t done = 0; done < size; done += GPU_WRITE_MAX_IO) {
            const int64_t chunk = std::min(GPU_WRITE_MAX_IO, size - done);
            const int64_t b = buf_off + done;
            if (b >= total_bytes) break;
            if (b + chunk > total_bytes) {
                const int64_t full = (total_bytes - b) / page_size * page_size;
                if (full > 0) ios.push_back({b, file_off + done, full, full});
                ios.push_back({b + full, file_off + done + full, page_size, total_bytes - b - full});
                continue;
            }
            ios.push_back({b, file_off + done, chunk, chunk});
        }
    }
    for (const WriteIo& io : ios) {
        TORCH_CHECK(io.dev_off % sector_size == 0 && io.size % sector_size == 0,
                    "gpu_write: unaligned I/O dev_off=", io.dev_off, " size=", io.size);
    }

    // Mark the point on the caller's stream where src is fully produced.
    cudaEvent_t ready;
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    C10_CUDA_CHECK(cudaEventRecord(ready, at::cuda::getCurrentCUDAStream(device)));

    auto job = [fd, device, ready, src, sector_size, ios = std::move(ios)]() {
        c10::cuda::CUDAGuard guard(device);
        C10_CUDA_CHECK(cudaEventSynchronize(ready));
        C10_CUDA_CHECK(cudaEventDestroy(ready));
        if (ios.empty()) return;

        auto stream = c10::cuda::getStreamFromPool(false, device);
        const auto pinned = torch::TensorOptions().dtype(torch::kUInt8).pinned_memory(true);
        torch::Tensor stage = torch::empty({NVME_WINDOW}, pinned);
        uint8_t* st = stage.data_ptr<uint8_t>();
        const uint8_t* src_raw = static_cast<const uint8_t*>(src.data_ptr());

        size_t next = 0;
        while (next < ios.size()) {
            const size_t first = next;
            int64_t used = 0;
            while (next < ios.size() && used + ios[next].size <= NVME_WINDOW) {
                used += ios[next].size;
                ++next;
            }
            const size_t n = next - first;

            std::vector<int64_t> stage_off(n);
            for (size_t i = 0, off = 0; i < n; ++i) {
                stage_off[i] = off;
                off += ios[first + i].size;
            }

            // GPU -> staging. Merge I/Os that are back to back in src (a tail
            // page ends a span, since its zero padding is not in src).
            for (size_t i = 0; i < n;) {
                int64_t len = ios[first + i].valid;
                size_t j = i + 1;
                while (j < n && ios[first + j - 1].valid == ios[first + j - 1].size &&
                       ios[first + j].buf_off == ios[first + i].buf_off + len) {
                    len += ios[first + j].valid;
                    ++j;
                }
                C10_CUDA_CHECK(cudaMemcpyAsync(st + stage_off[i], src_raw + ios[first + i].buf_off,
                                               len, cudaMemcpyDeviceToHost, stream));
                i = j;
            }
            C10_CUDA_CHECK(cudaStreamSynchronize(stream));
            for (size_t i = 0; i < n; ++i) {
                const WriteIo& io = ios[first + i];
                if (io.valid < io.size) std::memset(st + stage_off[i] + io.valid, 0, io.size - io.valid);
            }

            // Staging -> SSD.
            run_nvme_parallel(n, [&](size_t i) {
                const WriteIo& io = ios[first + i];
                nvme_write(fd, st + stage_off[i], io.dev_off / sector_size, io.size, sector_size);
            });
        }
    };

    std::lock_guard<std::mutex> lock(g_gpu_write_mutex);
    const int64_t id = g_gpu_write_next_id++;
    g_gpu_write_jobs.emplace(id, std::async(std::launch::async, std::move(job)));
    return id;
}

// Gathers and GPU writes share one job table, so either wait works for both.
void gather_wait(int64_t id) { gpu_write_wait(id); }

// Blocking version: submit and wait.
void gather_partitions_direct_raw(
    int64_t dev_handle_ptr,
    torch::Tensor dst,
    torch::Tensor runs,
    int64_t sector_size
) {
    gather_wait(gather_partitions_direct_raw_async(dev_handle_ptr, dst, runs, sector_size));
}


bool device_read_and_verify(
    int64_t fd,
    torch::Tensor expected_src,  
    torch::Tensor runs,
    int64_t sector_size,
    int64_t page_size
) {
    TORCH_CHECK(!expected_src.is_cuda() && expected_src.is_contiguous(),
                "expected_src must be contiguous CPU");
    TORCH_CHECK(runs.dim() == 2 && runs.size(1) == 3 && runs.scalar_type() == at::kLong);

    auto r = runs.contiguous();
    const int64_t* rp = r.data_ptr<int64_t>();
    const uint8_t* src_raw = static_cast<const uint8_t*>(expected_src.data_ptr());
    const int64_t total_bytes = expected_src.numel() * expected_src.element_size();
    const int64_t MAX_IO = 128 * 1024;
    std::vector<uint8_t> tail(page_size);

    /* 1. Read-back buffer, zero-filled so unread bytes show up as mismatches */
    torch::Tensor read_cpu = torch::zeros({total_bytes}, torch::kUInt8);
    uint8_t* dst_raw = read_cpu.data_ptr<uint8_t>();

    /* 2. Read back using the exact same run/chunk/tail layout as device_write */
    for(int64_t k = 0; k < r.size(0); ++k){
        const int64_t file_off = rp[3*k], buf_off = rp[3*k+1], size = rp[3*k+2];

        for(int64_t done = 0; done < size; done += MAX_IO){
            const int64_t chunk = std::min(MAX_IO, size - done);
            const int64_t b = buf_off + done;
            if (b >= total_bytes) break;
            uint8_t* data = dst_raw + b;

            if(b + chunk > total_bytes){
                const int64_t full = (total_bytes - b) / page_size * page_size;
                if (full > 0) {                      // read the full pages first
                    nvme_read(fd, data, (file_off + done) / sector_size, full, sector_size);
                }

                // last partial page: read the whole page, keep only the valid bytes
                std::fill(tail.begin(), tail.end(), 0);
                nvme_read(fd, tail.data(), (file_off + done + full) / sector_size,
                          page_size, sector_size);
                std::memcpy(data + full, tail.data(), total_bytes - b - full);
                continue;
            }
            nvme_read(fd, data, (file_off + done) / sector_size, chunk, sector_size);
        }
    }

    /* 3. Byte-for-byte comparison */
    int64_t diff_count = 0;
    int64_t first_diff_byte = -1;

    for (int64_t b = 0; b < total_bytes; ++b) {
        if (src_raw[b] != dst_raw[b]) {
            if (first_diff_byte == -1) {
                first_diff_byte = b;
            }
            diff_count++;
        }
    }

    if (diff_count > 0) {
        // Find which run holds the first bad byte so we can report its device location
        int64_t diff_run = -1, diff_dev_off = -1;
        for (int64_t k = 0; k < r.size(0); ++k) {
            const int64_t file_off = rp[3*k], buf_off = rp[3*k+1], size = rp[3*k+2];
            if (first_diff_byte >= buf_off && first_diff_byte < buf_off + size) {
                diff_run = k;
                diff_dev_off = file_off + (first_diff_byte - buf_off);
                break;
            }
        }

        printf("\n================ [VERIFICATION DIFF DETECTED] ================\n");
        printf(" Total bytes checked    : %ld\n", total_bytes);
        printf(" Total mismatched bytes: %ld / %ld\n", diff_count, total_bytes);
        printf(" First mismatch at byte : %ld\n", first_diff_byte);
        if (diff_run >= 0) {
            printf(" -> Run Index           : %ld\n", diff_run);
            printf(" -> Device byte offset  : %ld\n", diff_dev_off);
            printf(" -> LBA                 : %ld\n", diff_dev_off / sector_size);
            printf(" -> Byte within Page    : %ld\n", diff_dev_off % page_size);
        } else {
            printf(" -> Byte is not covered by any run (never written/read)\n");
        }
        printf("\n Hex Dump around first mismatch (Offset %ld):\n", first_diff_byte);
        
        int64_t start = std::max<int64_t>(0, first_diff_byte - 16);
        int64_t end = std::min<int64_t>(total_bytes, first_diff_byte + 32);

        printf(" Offset | Expected (Ground Truth)   | Read-back (From NVMe)\n");
        printf(" ---------------------------------------------------------\n");
        for (int64_t k = start; k < end; ++k) {
            printf(" %06ld | 0x%02x %s                      | 0x%02x\n", 
                  k, src_raw[k], (src_raw[k] == dst_raw[k]) ? "==" : "!=", dst_raw[k]);
        }
        printf("==============================================================\n\n");
        return false;
    }

    return true;
}



void scatter_partitions(int pid, torch::Tensor src,
                        std::vector<torch::Tensor> dsts,
                        std::vector<torch::Tensor> boundaries) {
  AT_ASSERTM(src.is_cuda(), "Source must be a CUDA tensor");
  for (auto &dst : dsts) {
    AT_ASSERTM(!dst.is_cuda(), "Destinations must be CPU tensors");
  }
  AT_ASSERTM(src.is_contiguous(), "Source must be contiguous");

  auto stream = at::cuda::getCurrentCUDAStream(src.get_device());
  AT_ASSERTM(stream != at::cuda::getDefaultCUDAStream(src.get_device()),
             "Async scatter requires a non-default CUDA stream");

  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half,src.scalar_type(), "scatter_partitions", [&] {
    getD2HPool().run([=] {
      auto src_data = src.data_ptr<scalar_t>();
      int64_t feat_dim = src.numel() / src.size(0);
      int64_t total_offset = 0;

      // First: D2H copy intra-partition gradient, then accumulate
      auto self_size = dsts[pid].size(0);

      // Clone previous values BEFORE overwriting (for accumulation)
      auto prev_values = dsts[pid].detach().clone();

      cudaMemcpyAsync(dsts[pid].data_ptr<scalar_t>(), src_data,
                      self_size * feat_dim * sizeof(scalar_t),
                      cudaMemcpyDeviceToHost, stream);

      // Wait for D2H to complete before accumulation.
      cudaStreamSynchronize(stream);

      // Accumulate: new_values += previous_values (write-back pattern)
      dsts[pid].add_(prev_values);
      total_offset += self_size;

      // Then: scatter boundary gradients with index_put_ accumulate
      for (size_t i = 0; i < dsts.size(); i++) {
        if ((int)i == pid)
          continue;
        auto index = boundaries[i];
        if (index.numel() == 0)
          continue;

        // Temporary buffer for D2H of this boundary slice
        auto slice = torch::empty({index.size(0), feat_dim},
                                  dsts[i].options());
        cudaMemcpyAsync(slice.data_ptr<scalar_t>(),
                        src_data + total_offset * feat_dim,
                        index.size(0) * feat_dim * sizeof(scalar_t),
                        cudaMemcpyDeviceToHost, stream);

        // Wait for D2H before accumulation
        cudaStreamSynchronize(stream);

        // Accumulate into destination partition at boundary indices
        dsts[i].index_put_({index}, slice, /*accumulate=*/true);
        total_offset += index.size(0);
      }

      AT_ASSERTM(total_offset == src.size(0),
                  "Scatter: copied size mismatch with source");
    });
  });
}


// python3 setup.py build_ext --inplace