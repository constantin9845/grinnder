#include "async_transfer.h"

#include <ATen/cuda/CUDAContext.h>

#include "../thread_pool.h"

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
#include <fcntl.h>
#include <unistd.h>
#include <vector>
#include <string>
#include <future>
#include <cstring>

// Global storage to keep track of active CUfile handles by file descriptor
static std::unordered_map<int, CUfileHandle_t> g_cufile_handles;

// Opens files with O_DIRECT and registers them with GDS cuFile
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

    // Track the handle globally so gather operations can reuse it
    g_cufile_handles[fd] = handle;
    fds.push_back(fd);
  }

  return fds;
}

// Deregisters GDS handles and closes open file descriptors
void close_files(const std::vector<int>& fds) {
  for (int fd : fds) {
    auto it = g_cufile_handles.find(fd);
    if (it != g_cufile_handles.end()) {
      cuFileHandleDeregister(it->second);
      g_cufile_handles.erase(it);
    }
    close(fd);
  }
}

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cufile.h>
#include <fcntl.h>
#include <unistd.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <cstring>
#include <cstdlib>

#define MAX_BATCH_SIZE 256

extern std::unordered_map<int, CUfileHandle_t> g_cufile_handles;

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
    CUfileHandle_t target_handle = g_cufile_handles[target_fd];

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
      CUfileHandle_t handle = g_cufile_handles[fd];

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


extern std::unordered_map<int, CUfileHandle_t> g_cufile_handles2;

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



extern std::unordered_map<int, CUfileHandle_t> g_cufile_handles3;

void device_write(
    int raw_nvme_fd,
    torch::Tensor src,
    torch::Tensor ppa_list,
    int sector_size
){

    AT_ASSERTM(src.is_cuda(), "Source tensor must be on CUDA device");
    AT_ASSERTM(src.is_contiguous(), "Source tensor must be contiguous");
    AT_ASSERTM(ppa_list.is_contiguous(), "PPA tensor must be contiguous");

    auto stream = at::cuda::getCurrentCUDAStream(src.get_device());
    c10::cuda::CUDAStreamGuard guard(stream);

    CUfileHandle_t nvme_handle = g_cufile_handles3[raw_nvme_fd];
    uint8_t* src_raw = reinterpret_cast<uint8_t*>(src.data_ptr());

    int64_t total_bytes = src.numel() * src.element_size();
    int64_t num_sectors = ppa_list.numel();

    const int64_t* ppa_ptr = ppa_list.data_ptr<int64_t>();

    CUfileIOParams_t *io_params = (CUfileIOParams_t*)calloc(num_sectors, sizeof(CUfileIOParams_t));
    AT_ASSERTM(io_params != nullptr, "Failed to allocate io_params");

    for (int64_t i = 0; i < num_sectors; ++i) {
        uint64_t raw_ppa = static_cast<uint64_t>(ppa_ptr[i]);

        int64_t dev_offset = i * sector_size;
        int64_t write_bytes = std::min<int64_t>(sector_size, total_bytes - dev_offset);

        io_params[i].mode = CUFILE_BATCH;
        io_params[i].opcode = CUFILE_WRITE;
        io_params[i].fh = nvme_handle;
        io_params[i].u.batch.devPtr_base = src_raw;
        io_params[i].u.batch.devPtr_offset = dev_offset;
        
        // Encode 64-bit PPA as byte offset for NVMe driver
        io_params[i].u.batch.file_offset = static_cast<off_t>(raw_ppa * sector_size);
        io_params[i].u.batch.size = write_bytes;
    }

    CUfileBatchHandle_t batch_handle;
    CUfileError_t status = cuFileBatchIOSetUp(&batch_handle, MAX_BATCH_SIZE);
    AT_ASSERTM(status.err == CU_FILE_SUCCESS, "cuFileBatchIOSetUp failed");

    CUfileIOEvents_t events[MAX_BATCH_SIZE];

    for (size_t offset_idx = 0; offset_idx < num_sectors; offset_idx += MAX_BATCH_SIZE) {
        unsigned int current_batch_size = (num_sectors - offset_idx > MAX_BATCH_SIZE)
                                         ? MAX_BATCH_SIZE
                                         : (unsigned int)(num_sectors - offset_idx);

        status = cuFileBatchIOSubmit(batch_handle, current_batch_size, &io_params[offset_idx], 0);
        AT_ASSERTM(status.err == CU_FILE_SUCCESS, "cuFileBatchIOSubmit write failed");

        unsigned int completed = current_batch_size;
        status = cuFileBatchIOGetStatus(batch_handle, current_batch_size, &completed, events, NULL);
        AT_ASSERTM(status.err == CU_FILE_SUCCESS, "cuFileBatchIOGetStatus write failed");
    }

    cuFileBatchIODestroy(batch_handle);
    free(io_params);

}


extern std::unordered_map<int, CUfileHandle_t> g_cufile_handles4;

bool device_read_and_verify(
    int raw_nvme_fd,
    torch::Tensor expected_src,  /* Ground truth tensor (on CUDA) */
    torch::Tensor ppa_list,      /* Tensor of uint64 PPAs */
    int sector_size
){
  AT_ASSERTM(expected_src.is_cuda(), "Expected source tensor must be on CUDA device");
    AT_ASSERTM(expected_src.is_contiguous(), "Expected source tensor must be contiguous");
    AT_ASSERTM(ppa_list.is_contiguous(), "PPA tensor must be contiguous");

    auto stream = at::cuda::getCurrentCUDAStream(expected_src.get_device());
    c10::cuda::CUDAStreamGuard guard(stream);

    CUfileHandle_t nvme_handle = g_cufile_handles3[raw_nvme_fd];

    int64_t total_bytes = expected_src.numel() * expected_src.element_size();
    int64_t num_sectors = ppa_list.numel();

    /* 1. Allocate a destination GPU buffer matching expected_src size and device */
    torch::Tensor read_dst = torch::empty_like(expected_src);
    uint8_t* dst_raw = reinterpret_cast<uint8_t*>(read_dst.data_ptr());

    const int64_t* ppa_ptr = ppa_list.data_ptr<int64_t>();

    CUfileIOParams_t *io_params = (CUfileIOParams_t*)calloc(num_sectors, sizeof(CUfileIOParams_t));
    AT_ASSERTM(io_params != nullptr, "Failed to allocate io_params");


    for (int64_t i = 0; i < num_sectors; ++i) {
        uint64_t raw_ppa = static_cast<uint64_t>(ppa_ptr[i]);

        int64_t dev_offset = i * sector_size;
        int64_t read_bytes = std::min<int64_t>(sector_size, total_bytes - dev_offset);

        io_params[i].mode = CUFILE_BATCH;
        io_params[i].opcode = CUFILE_READ; /* Reading back from NVMe to GPU */
        io_params[i].fh = nvme_handle;
        io_params[i].u.batch.devPtr_base = dst_raw;
        io_params[i].u.batch.devPtr_offset = dev_offset;
        
        /* Send raw PPA encoded for NVMe backend translation */
        io_params[i].u.batch.file_offset = static_cast<off_t>(raw_ppa * sector_size);
        io_params[i].u.batch.size = read_bytes;
    }

    CUfileBatchHandle_t batch_handle;
    CUfileError_t status = cuFileBatchIOSetUp(&batch_handle, MAX_BATCH_SIZE);
    AT_ASSERTM(status.err == CU_FILE_SUCCESS, "cuFileBatchIOSetUp failed");

    CUfileIOEvents_t events[MAX_BATCH_SIZE];

    for (size_t offset_idx = 0; offset_idx < num_sectors; offset_idx += MAX_BATCH_SIZE) {
        unsigned int current_batch_size = (num_sectors - offset_idx > MAX_BATCH_SIZE)
                                         ? MAX_BATCH_SIZE
                                         : (unsigned int)(num_sectors - offset_idx);

        status = cuFileBatchIOSubmit(batch_handle, current_batch_size, &io_params[offset_idx], 0);
        AT_ASSERTM(status.err == CU_FILE_SUCCESS, "cuFileBatchIOSubmit read failed");

        unsigned int completed = current_batch_size;
        status = cuFileBatchIOGetStatus(batch_handle, current_batch_size, &completed, events, NULL);
        AT_ASSERTM(status.err == CU_FILE_SUCCESS, "cuFileBatchIOGetStatus read failed");

        /* Validate individual batch status results */
        for (unsigned int e = 0; e < completed; ++e) {
            AT_ASSERTM(events[e].status == CU_FILE_SUCCESS, "cuFile batch IO read event error");
        }
    }

    cuFileBatchIODestroy(batch_handle);
    free(io_params);

    /* 4. Ensure CUDA stream completes before verification */
    stream.synchronize();

    /* 5. Compare written source tensor with read-back target tensor */
    bool is_equal = torch::equal(expected_src, read_dst);
    return is_equal;
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