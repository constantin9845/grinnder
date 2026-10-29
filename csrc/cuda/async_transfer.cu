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
        io_params[param_idx].u.batch.devPtr_base = dst_raw;
        io_params[param_idx].u.batch.devPtr_offset = (start_row_offset + idx) * row_bytes;
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

void gather_partitions_direct_raw(
    int64_t dev_handle_ptr,
    torch::Tensor dst,
    torch::Tensor runs,          /* [N, 3] int64: (dev_off, dst_off, size), same layout as device_write */
    int64_t sector_size
) {
    AT_ASSERTM(dst.is_cuda(), "Destination must be a CUDA tensor");
    AT_ASSERTM(dst.is_contiguous(), "Destination must be contiguous");
    TORCH_CHECK(runs.dim() == 2 && runs.size(1) == 3 && runs.scalar_type() == at::kLong,
                "runs must be an [N, 3] int64 tensor");

    CUfileHandle_t dev_handle =
        reinterpret_cast<CUfileHandle_t>(static_cast<uintptr_t>(dev_handle_ptr));

    auto r = runs.to(at::kCPU).contiguous();
    const int64_t* rp = r.data_ptr<int64_t>();
    void* dst_raw = dst.data_ptr();
    const int64_t dst_bytes = dst.numel() * dst.element_size();
    const int64_t MAX_IO = 128 * 1024;

    for (int64_t k = 0; k < r.size(0); ++k) {
        const int64_t dev_off = rp[3*k], buf_off = rp[3*k+1], size = rp[3*k+2];

        TORCH_CHECK(dev_off % sector_size == 0 && size % sector_size == 0,
                    "gather run ", k, " not sector-aligned: dev_off=", dev_off, " size=", size);
        TORCH_CHECK(buf_off >= 0 && buf_off + size <= dst_bytes,
                    "gather run ", k, " exceeds dst: buf_off=", buf_off,
                    " size=", size, " dst_bytes=", dst_bytes);

        for (int64_t done = 0; done < size; done += MAX_IO) {
            const int64_t chunk = std::min(MAX_IO, size - done);

            const ssize_t ret = cuFileRead(
                dev_handle,
                dst_raw,              // devPtr_base
                chunk,
                dev_off + done,       // device byte offset (LBA * sector_size)
                buf_off + done        // devPtr_offset into dst
            );

            TORCH_CHECK(ret == chunk,
                        "cuFileRead failed: run=", k,
                        ", dev_off=", dev_off + done,
                        ", lba=", (dev_off + done) / sector_size,
                        ", dst_off=", buf_off + done,
                        ", bytes=", chunk,
                        ", returned=", ret,
                        ", errno=", errno, " (", strerror(errno), ")");
        }
    }

    // Make sure the data is in GPU memory before returning.
    C10_CUDA_CHECK(cudaDeviceSynchronize());
}


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



void gpu_write(
  int64_t fd,
  torch::Tensor src,
  torch::Tensor ppa_list,
  int sector_size
){
  TORCH_CHECK(src.is_cuda(), "Source tensor must be on GPU");
  TORCH_CHECK(src.is_contiguous(), "Source tensor must be contiguous");
  TORCH_CHECK(ppa_list.is_contiguous(), "PPA tensor must be contiguous");


  auto ppa_cpu = ppa_list.to(at::kCPU).contiguous();
  const int64_t* ppa_ptr = ppa_cpu.data_ptr<int64_t>();
  int64_t total_bytes = src.numel() * src.element_size();
  int64_t num_pages = ppa_list.numel();
  int64_t page_bytes = static_cast<int64_t>(sector_size) * 32;

  CUfileHandle_t handle = reinterpret_cast<CUfileHandle_t>(static_cast<uintptr_t>(fd));
  const void* src_raw = src.data_ptr();

  for (int64_t i = 0; i < num_pages; ++i){
    uint64_t raw_ppa = static_cast<uint64_t>(ppa_ptr[i]);
    // encode host ftl bit
    raw_ppa = encode_host_ftl_bit(raw_ppa);
    int64_t dev_offset = i * page_bytes;
    int64_t write_bytes = std::min<int64_t>(page_bytes, total_bytes - dev_offset);
    if (write_bytes <= 0) break;

    TORCH_CHECK(write_bytes % sector_size == 0, "Write size must be sector-aligned");

    ssize_t ret = cuFileWrite(
        handle,
        static_cast<const char*>(src_raw) + dev_offset,
        write_bytes,
        static_cast<int64_t>(raw_ppa),
        dev_offset
    );

    if (ret < 0) {
        TORCH_CHECK(
            false,
            "cuFileWrite failed: page=", i,
            ", ret=", ret,
            ", errno=", errno,
            " (", strerror(errno), ")",
            ", handle=", (void*)handle,
            ", bytes=", write_bytes,
            ", file_offset=", raw_ppa,
            ", buffer_offset=", dev_offset
        );
    }
  }

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