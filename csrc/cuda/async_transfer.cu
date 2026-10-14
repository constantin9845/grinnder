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

/*
void gather_partitions_direct_raw(
  int pid,
  int64_t dev_handle_ptr,                                   
  torch::Tensor dst,
  std::vector<torch::Tensor> boundaries,
  std::vector<torch::Tensor> boundary_ppas,       
  std::vector<torch::Tensor> boundary_page_offs,  
  std::vector<torch::Tensor> boundary_row_offsets, 
  int rows_per_page,
  bool feature_type,
  int64_t row_bytes
){

  AT_ASSERTM(dst.is_cuda(), "Destination must be a CUDA tensor");
  AT_ASSERTM(dst.is_contiguous(), "Destination must be contiguous");
  TORCH_CHECK(row_bytes > 0, "row_bytes must be positive");

  const size_t num_parts = boundary_ppas.size();

  TORCH_CHECK(pid >= 0 && static_cast<size_t>(pid) < num_parts, "Invalid partition ID");
  TORCH_CHECK(boundaries.size() == num_parts, "boundaries and boundary_ppas size mismatch");
  TORCH_CHECK(boundary_row_offsets.size() == num_parts, "boundary_row_offsets and boundary_ppas size mismatch");

  (void)boundary_page_offs;
  CUfileHandle_t dev_handle = reinterpret_cast<CUfileHandle_t>(static_cast<uintptr_t>(dev_handle_ptr));

  auto stream = at::cuda::getCurrentCUDAStream(dst.get_device());

  getH2DPool().run([=] {

    c10::cuda::CUDAStreamGuard guard(stream);

    uint8_t* dst_raw = reinterpret_cast<uint8_t*>(dst.data_ptr());

    // rows from other partitions
    size_t boundary_reads = 0;

    for (size_t i = 0; i < num_parts; ++i) {
        if (static_cast<int>(i) != pid &&
            boundaries[i].defined() &&
            boundaries[i].numel() > 0) {
            boundary_reads += boundaries[i].numel();
        }
    }

    const int64_t target_nodes = boundaries[pid].defined() ? boundaries[pid].numel() : (dst.size(0) - static_cast<int64_t>(boundary_reads));

    TORCH_CHECK(target_nodes >= 0, "Invalid target row count");
    TORCH_CHECK(
        target_nodes + static_cast<int64_t>(boundary_reads)
            <= dst.size(0),
        "Requested rows exceed destination tensor size"
    );

    
    // ---------------------------------------------------------
    // 1. Full target partition
    // ---------------------------------------------------------
    if(target_nodes > 0){
      TORCH_CHECK(boundary_ppas[pid].defined(),
                  "Target PPAs are undefined");
      TORCH_CHECK(boundary_row_offsets[pid].defined(),
                  "Target row offsets are undefined");


      auto target_ppas = boundary_ppas[pid].to(at::kCPU).contiguous();
      auto target_rows = boundary_row_offsets[pid].to(at::kCPU).contiguous();

      TORCH_CHECK(target_ppas.numel() >= target_nodes,
                  "Not enough target PPAs");
      TORCH_CHECK(target_rows.numel() >= target_nodes,
                  "Not enough target row offsets");

      const uint64_t* ppa_ptr = target_ppas.data_ptr<uint64_t>();
      const int64_t* row_ptr = target_rows.data_ptr<int64_t>();

      for (int64_t r = 0; r<target_nodes; ++r){
        uint64_t raw_ppa = ppa_ptr[r];
        // encode host ftl bit
        raw_ppa = encode_host_ftl_bit(raw_ppa);
        const int64_t row_offset = row_ptr[r];

        printf("Type bit = %d\n", feature_type);
        const uint64_t request_ppa = encode_row_request_ppa(rows_per_page, raw_ppa, row_offset, feature_type);

        const int64_t dst_byte_offset = r * row_bytes;

        // FEMU will return only this row (not full page)
        // add batch io
        const ssize_t ret = cuFileRead(
            dev_handle,
            dst_raw + dst_byte_offset,
            row_bytes,
            static_cast<int64_t>(request_ppa),
            0
        );

        TORCH_CHECK(
            ret == row_bytes,
            "GDS target row read failed: row=", r,
            ", returned=", ret,
            ", expected=", row_bytes
        );
      }
    }

    // ---------------------------------------------------------
    // 2. Read boundary rows 
    // ---------------------------------------------------------

    if (boundary_reads == 0) {
      return;
    }

    std::vector<CUfileIOParams_t> io_params(boundary_reads);
    size_t param_idx = 0;
    int64_t curr_dst_row = target_nodes;

    for (size_t i = 0; i < num_parts; ++i){

      if (static_cast<int>(i) == pid ||
        !boundaries[i].defined() ||
        boundaries[i].numel() == 0){
        continue;
      }

      TORCH_CHECK(boundary_ppas[i].defined(),
                  "Boundary PPAs are undefined for partition ", i);
      TORCH_CHECK(boundary_row_offsets[i].defined(),
                  "Boundary row offsets are undefined for partition ", i);


      auto ppas = boundary_ppas[i].to(at::kCPU).contiguous();
      auto rows = boundary_row_offsets[i].to(at::kCPU).contiguous();
      const int64_t num_rows = boundaries[i].numel();

      TORCH_CHECK(ppas.numel() >= num_rows,
                  "Not enough boundary PPAs for partition ", i);
      TORCH_CHECK(rows.numel() >= num_rows,
                  "Not enough boundary row offsets for partition ", i);

      const uint64_t* ppa_ptr = ppas.data_ptr<uint64_t>();
      const int64_t* row_ptr = rows.data_ptr<int64_t>();

      for (int64_t r = 0; r < num_rows; ++r){
        uint64_t raw_ppa = ppa_ptr[r];
        // encode host ftl bit
        raw_ppa = encode_host_ftl_bit(raw_ppa);
        const int64_t row_offset = row_ptr[r];

        const uint64_t request_ppa = encode_row_request_ppa(rows_per_page, raw_ppa, row_offset, feature_type);
        const int64_t dst_byte_offset = curr_dst_row * row_bytes;

        CUfileIOParams_t& param = io_params[param_idx];

        param.mode = CUFILE_BATCH;
        param.opcode = CUFILE_READ;
        param.fh = dev_handle;
        param.u.batch.devPtr_base = dst_raw;
        param.u.batch.devPtr_offset = dst_byte_offset;
        param.u.batch.file_offset = static_cast<int64_t>(request_ppa); // raw ppa + in-page offset
        param.u.batch.size = row_bytes;

        ++curr_dst_row;
        ++param_idx;
      }
    }

    TORCH_CHECK(param_idx == boundary_reads, "Internal boundary read count mismatch");

    CUfileBatchHandle_t batch_handle;
    CUfileError_t status = cuFileBatchIOSetUp(&batch_handle, MAX_BATCH_SIZE);
    TORCH_CHECK(status.err == CU_FILE_SUCCESS, "cuFileBatchIOSetUp failed");
    CUfileIOEvents_t events[MAX_BATCH_SIZE];

    for (size_t offset_idx = 0; offset_idx < boundary_reads; offset_idx += MAX_BATCH_SIZE){
      const unsigned int current_batch_size = static_cast<unsigned int>(std::min<size_t>(MAX_BATCH_SIZE,boundary_reads - offset_idx));

      status = cuFileBatchIOSubmit(batch_handle,current_batch_size,io_params.data() + offset_idx,0);

      TORCH_CHECK(status.err == CU_FILE_SUCCESS, "cuFileBatchIOSubmit failed");

      unsigned int completed = 0;

      status = cuFileBatchIOGetStatus(
          batch_handle,
          current_batch_size,
          &completed,
          events,
          nullptr
      );

      TORCH_CHECK(status.err == CU_FILE_SUCCESS, "cuFileBatchIOGetStatus failed");
      TORCH_CHECK(completed == current_batch_size, "Not all boundary reads completed");
    }

    cuFileBatchIODestroy(batch_handle);
  });

  C10_CUDA_CHECK(cudaGetLastError());
}
*/

void gather_partitions_direct_raw(
    int pid,
    int64_t dev_handle_ptr,
    torch::Tensor dst,
    std::vector<torch::Tensor> boundaries,
    std::vector<torch::Tensor> boundary_ppas,
    std::vector<torch::Tensor> boundary_page_offs,
    std::vector<torch::Tensor> boundary_row_offsets,
    int rows_per_page,
    bool feature_type,
    int64_t row_bytes
) {
    AT_ASSERTM(dst.is_cuda(), "Destination must be a CUDA tensor");
    AT_ASSERTM(dst.is_contiguous(), "Destination must be contiguous");
    TORCH_CHECK(row_bytes > 0, "row_bytes must be positive");

    const size_t num_parts = boundary_ppas.size();

    (void)boundary_page_offs;

    CUfileHandle_t dev_handle =
        reinterpret_cast<CUfileHandle_t>(
            static_cast<uintptr_t>(dev_handle_ptr)
        );

    // No thread pool: execute everything on the calling thread.
    uint8_t* dst_raw =
        reinterpret_cast<uint8_t*>(dst.data_ptr());

    // ---------------------------------------------------------
    // Count boundary rows belonging to other partitions.
    // ---------------------------------------------------------
    size_t boundary_reads = 0;

    for (size_t i = 0; i < num_parts; ++i) {
        if (static_cast<int>(i) != pid &&
            boundaries[i].defined() &&
            boundaries[i].numel() > 0) {
            boundary_reads += boundaries[i].numel();
        }
    }

    const int64_t target_nodes =
        boundaries[pid].defined()
            ? boundaries[pid].numel()
            : (dst.size(0) - static_cast<int64_t>(boundary_reads));

    TORCH_CHECK(target_nodes >= 0, "Invalid target row count");
    TORCH_CHECK(
        target_nodes + static_cast<int64_t>(boundary_reads)
            <= dst.size(0),
        "Requested rows exceed destination tensor size"
    );

    printf(
        "[GDS] pid=%d target_nodes=%lld boundary_reads=%zu\n",
        pid,
        (long long)target_nodes,
        boundary_reads
    );
    fflush(stdout);

    // ---------------------------------------------------------
    // 1. Read rows from the target partition.
    // ---------------------------------------------------------
    if (target_nodes > 0) {


        auto target_ppas =
            boundary_ppas[pid].to(at::kCPU).contiguous();
        auto target_rows =
            boundary_row_offsets[pid].to(at::kCPU).contiguous();


        const int64_t* ppa_ptr = target_ppas.data_ptr<int64_t>();
        const int64_t* row_ptr =
            target_rows.data_ptr<int64_t>();

        for (int64_t r = 0; r < target_nodes; ++r) {
            uint64_t raw_ppa = ppa_ptr[r];
            raw_ppa = encode_host_ftl_bit(raw_ppa);

            const int64_t row_offset = row_ptr[r];

            printf("Type bit = %d\n", feature_type);

            const uint64_t request_ppa =
                encode_row_request_ppa(
                    rows_per_page,
                    raw_ppa,
                    row_offset,
                    feature_type
                );

            const int64_t dst_byte_offset = r * row_bytes;

            printf(
                "[GDS READ BEFORE] row=%ld request_ppa=0x%016lx "
                "file_offset=%ld bytes=%ld\n",
                r,
                static_cast<unsigned long>(request_ppa),
                row_offset,
                row_bytes
            );
            fflush(stdout);

            const ssize_t ret = cuFileRead(
                dev_handle,
                dst_raw + dst_byte_offset,
                row_bytes,
                static_cast<int64_t>(request_ppa),
                0
            );

            printf(
                "[GDS READ AFTER] row=%ld returned=%ld errno=%d\n",
                r, static_cast<long>(ret), errno
            );
            fflush(stdout);

            TORCH_CHECK(
                ret == row_bytes,
                "GDS target row read failed: row=", r,
                ", returned=", ret,
                ", expected=", row_bytes,
                ", errno=", errno,
                " (", strerror(errno), ")"
            );
        }
    }

    // ---------------------------------------------------------
    // 2. Read boundary rows sequentially.
    // ---------------------------------------------------------
    if (boundary_reads > 0) {
        int64_t curr_dst_row = target_nodes;

        for (size_t i = 0; i < num_parts; ++i) {
            if (static_cast<int>(i) == pid ||
                !boundaries[i].defined() ||
                boundaries[i].numel() == 0) {
                continue;
            }


            auto ppas =
                boundary_ppas[i].to(at::kCPU).contiguous();
            auto rows =
                boundary_row_offsets[i].to(at::kCPU).contiguous();

            const int64_t num_rows = boundaries[i].numel();


            const int64_t* ppa_ptr = ppas.data_ptr<int64_t>();
            const int64_t* row_ptr =
                rows.data_ptr<int64_t>();

            for (int64_t r = 0; r < num_rows; ++r) {
                uint64_t raw_ppa = ppa_ptr[r];
                raw_ppa = encode_host_ftl_bit(raw_ppa);

                const int64_t row_offset = row_ptr[r];

                const uint64_t request_ppa =
                    encode_row_request_ppa(
                        rows_per_page,
                        raw_ppa,
                        row_offset,
                        feature_type
                    );

                const int64_t dst_byte_offset =
                    curr_dst_row * row_bytes;

                printf(
                    "[GDS BOUNDARY] partition=%zu row=%lld "
                    "dst_row=%lld ppa=0x%016llx "
                    "row_offset=%lld bytes=%lld\n",
                    i,
                    (long long)r,
                    (long long)curr_dst_row,
                    (unsigned long long)request_ppa,
                    (long long)row_offset,
                    (long long)row_bytes
                );
                fflush(stdout);

                const ssize_t ret = cuFileRead(
                    dev_handle,
                    dst_raw + dst_byte_offset,
                    row_bytes,
                    static_cast<int64_t>(request_ppa),
                    0
                );

                TORCH_CHECK(
                    ret == row_bytes,
                    "GDS boundary row read failed: partition=", i,
                    ", row=", r,
                    ", returned=", ret,
                    ", expected=", row_bytes,
                    ", errno=", errno,
                    " (", strerror(errno), ")"
                );

                ++curr_dst_row;
            }
        }

        TORCH_CHECK(
            curr_dst_row ==
                target_nodes + static_cast<int64_t>(boundary_reads),
            "Internal boundary read count mismatch"
        );
    }

    // Ensure GPU operations on the current device are complete
    // before returning to the caller.
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


#include <sys/ioctl.h>
#include <linux/nvme_ioctl.h>

void device_write(
    int64_t fd,
    torch::Tensor src,
    torch::Tensor ppa_list,
    int sector_size
) {
    TORCH_CHECK(!src.is_cuda(), "Source tensor must be on CPU for POSIX writes");
    TORCH_CHECK(src.is_contiguous(), "Source tensor must be contiguous");
    TORCH_CHECK(ppa_list.is_contiguous(), "PPA tensor must be contiguous");

    const uint8_t* src_raw = reinterpret_cast<const uint8_t*>(src.data_ptr());
    const int64_t* ppa_ptr = ppa_list.data_ptr<int64_t>();

    int64_t total_bytes = src.numel() * src.element_size();
    int64_t num_pages = ppa_list.numel();
    int raw_fd = static_cast<int>(fd);

    for (int64_t i = 0; i < num_pages; ++i) {
        uint64_t raw_ppa = static_cast<uint64_t>(ppa_ptr[i]);

        // encode host ftl bit
        raw_ppa = encode_host_ftl_bit(raw_ppa);

        int64_t dev_offset = i * (sector_size*32);
        int64_t write_bytes = std::min<int64_t>((sector_size*32), total_bytes - dev_offset);

        if (write_bytes <= 0) break;

        int64_t sectors = (write_bytes + sector_size - 1) / sector_size;

        struct nvme_user_io io {};
        std::memset(&io, 0, sizeof(io));
        io.opcode = 0x01; 
        io.addr = reinterpret_cast<uint64_t>(src_raw + dev_offset);
        io.slba = raw_ppa;
        io.nblocks = sectors-1;

        int ret = ioctl(raw_fd, NVME_IOCTL_SUBMIT_IO, &io);

        if (ret < 0) {
            TORCH_CHECK(
                false,
                "NVME_IOCTL_SUBMIT_IO write failed at sector ", i,
                " (PPA: ", raw_ppa, "). Error: ", strerror(errno),
                " (errno ", errno, ")"
            );
        }
        else if(ret > 0){
          printf("ret = %d\n", ret);
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
    torch::Tensor expected_src,  /* Ground truth CPU tensor */
    torch::Tensor ppa_list,      /* Tensor of uint64 PPAs */
    int sector_size
) {
    TORCH_CHECK(!expected_src.is_cuda(), "expected_src tensor must be on CPU");
    TORCH_CHECK(expected_src.is_contiguous(), "expected_src tensor must be contiguous");
    TORCH_CHECK(ppa_list.is_contiguous(), "PPA tensor must be contiguous");

    int64_t total_bytes = expected_src.numel() * expected_src.element_size();
    int64_t num_pages = ppa_list.numel();
    int raw_fd = static_cast<int>(fd);

    /* 1. Allocate a CPU destination buffer matching expected_src dimensions/dtype */
    torch::Tensor read_cpu = torch::zeros(
        expected_src.sizes(),
        torch::TensorOptions().dtype(expected_src.dtype()).device(torch::kCPU)
    );
    
    uint8_t* dst_raw = reinterpret_cast<uint8_t*>(read_cpu.data_ptr());
    const uint8_t* src_raw = reinterpret_cast<const uint8_t*>(expected_src.data_ptr());
    const int64_t* ppa_ptr = ppa_list.data_ptr<int64_t>();

    /* 2. Read sectors via NVMe IOCTL opcode 0x02 (nvme_cmd_read) */
    for (int64_t i = 0; i < num_pages; ++i) {
        uint64_t raw_ppa = static_cast<uint64_t>(ppa_ptr[i]);
        // encode host ftl bit
        raw_ppa = encode_host_ftl_bit(raw_ppa);

        int64_t dev_offset = i * (sector_size*32);
        int64_t read_bytes = std::min<int64_t>((sector_size*32), total_bytes - dev_offset);

        if (read_bytes <= 0) break;

        int64_t sectors = (read_bytes + sector_size - 1) / sector_size;

        struct nvme_user_io io {};
        std::memset(&io, 0, sizeof(io));

        io.opcode = 0x02; // nvme_cmd_read
        io.flags = 0;
        io.control = 0;
        io.metadata = 0;
        
        // Target buffer pointer inside read_cpu
        io.addr = reinterpret_cast<uint64_t>(dst_raw + dev_offset);
        
        // Pass OCSSD PPA directly to cdw10/cdw11
        io.slba = raw_ppa;
        
        // NVMe nblocks is 0-based
        io.nblocks = sectors-1;

        int ret = ioctl(raw_fd, NVME_IOCTL_SUBMIT_IO, &io);

        if (ret != 0) {
            TORCH_CHECK(
                false,
                "NVME_IOCTL_SUBMIT_IO read failed at sector ", i,
                " (PPA: ", raw_ppa, "). Return code: ", ret,
                " | System Error: ", strerror(errno), " (errno ", errno, ")"
            );
        }
    }

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
        int64_t diff_page_idx = first_diff_byte / (sector_size*32);
        int64_t byte_in_page = first_diff_byte % (sector_size*32);
        uint64_t diff_ppa = static_cast<uint64_t>(ppa_ptr[diff_page_idx]);

        printf("\n================ [VERIFICATION DIFF DETECTED] ================\n");
        printf(" Total bytes checked    : %ld\n", total_bytes);
        printf(" Total mismatched bytes: %ld / %ld\n", diff_count, total_bytes);
        printf(" First mismatch at byte : %ld\n", first_diff_byte);
        printf(" -> Page Index          : %ld\n", diff_page_idx);
        printf(" -> Byte within Page    : %ld\n", byte_in_page);
        printf(" -> Target PPA          : 0x%016lx\n", diff_ppa);
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

    /* 3. Perform byte-for-byte CPU tensor comparison */
    return torch::equal(expected_src, read_cpu);
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