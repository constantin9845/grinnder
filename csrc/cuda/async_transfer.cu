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
        // Unregistered buffer: GDS needs devPtr_offset + size <= 1 MiB, so
        // fold the offset into the pointer (see gpu_write_run_batches).
        io_params[param_idx].u.batch.devPtr_base =
            dst_raw + (start_row_offset + idx) * row_bytes;
        io_params[param_idx].u.batch.devPtr_offset = 0;
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
// Only one cuFile batch in flight at a time. With real GDS (not compat mode),
// overlapping batches from the gather and write threads hit an internal
// libcufile assertion (cufio_batch line 521).
static std::mutex g_cufile_batch_mutex;
static std::unordered_map<int64_t, std::future<void>> g_gpu_write_jobs;
static int64_t g_gpu_write_next_id = 0;

// Runs every I/O in `ios` (CUFILE_READ or CUFILE_WRITE) through the cuFile
// batch API, GPU_WRITE_BATCH entries at a time. Shared by GPU writes and gathers.
static void gpu_write_run_batches(CUfileHandle_t handle, const std::vector<GpuWriteIo>& ios,
                                  CUfileOpcode_t opcode) {
    if (ios.empty()) return;
    const char* what = opcode == CUFILE_READ ? "read" : "write";
    std::lock_guard<std::mutex> batch_lock(g_cufile_batch_mutex);

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
            // GPU buffers are not cuFileBufRegister'ed. For unregistered memory the
            // GDS path stages each IO through a 1 MiB internal buffer and rejects
            // devPtr_offset + size > 1 MiB, so put the offset into the pointer.
            params[i].u.batch.devPtr_base = static_cast<char*>(io.base) + io.buf_off;
            params[i].u.batch.devPtr_offset = 0;
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