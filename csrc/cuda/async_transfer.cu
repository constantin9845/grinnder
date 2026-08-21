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


#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cufile.h>
#include <fcntl.h>
#include <unistd.h>
#include <vector>
#include <string>

void gather_partitions_gds(
    int pid,
    std::vector<std::string> file_paths,
    std::vector<int64_t> num_nodes,
    torch::Tensor dst,
    std::vector<torch::Tensor> boundaries
) {
  AT_ASSERTM(dst.is_cuda(), "Destination must be a CUDA tensor");
  AT_ASSERTM(dst.is_contiguous(), "Destination must be contiguous");

  const int device_id = dst.get_device();
  c10::cuda::CUDAGuard device_guard(device_id);

  auto stream = at::cuda::getCurrentCUDAStream(device_id);
  cudaStream_t raw_stream = stream.stream();

  AT_ASSERTM(stream != at::cuda::getDefaultCUDAStream(device_id), "Async gather requires a non-default CUDA stream");

  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, dst.scalar_type(), "gather_partitions_gds", [&] {
    getH2DPool().run([=] {
      auto dst_data = reinterpret_cast<uint8_t*>(dst.data_ptr<scalar_t>());
      int64_t feat_dim = dst.numel() / dst.size(0);
      int64_t row_bytes = feat_dim * sizeof(scalar_t);

      // 1. Read full target partition
      int64_t offset = num_nodes[pid];
      int fd_target = open(file_paths[pid].c_str(), O_RDONLY | O_DIRECT);
      AT_ASSERTM(fd_target >= 0, "GDS Gather: Failed to open target partition file");

      CUfileDescr_t descr_target;
      memset(&descr_target, 0, sizeof(descr_target));
      descr_target.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
      descr_target.handle.fd = fd_target;

      CUfileHandle_t cf_target;
      CUfileError_t status = cuFileHandleRegister(&cf_target, &descr_target);
      AT_ASSERTM(status.err == CU_FILE_SUCCESS, "GDS Gather: cuFileHandleRegister failed for target file");

      // Set variables matching cuFileReadAsync exact types
      size_t target_size = static_cast<size_t>(offset * row_bytes);
      off_t target_file_offset = 0;
      off_t target_buf_offset = 0;
      ssize_t target_bytes_read = 0;

      cuFileReadAsync(cf_target, dst_data, &target_size, &target_file_offset, &target_buf_offset, &target_bytes_read, raw_stream);

      // 2. Read boundary partition chunks
      for (size_t i = 0; i < file_paths.size(); i++) {
        if ((int)i == pid)
          continue;

        auto bndry = boundaries[i];
        if (bndry.numel() == 0)
          continue;

        auto bndry_cpu = bndry.to(torch::kCPU, torch::kLong).contiguous();
        const int64_t* idx_ptr = bndry_cpu.data_ptr<int64_t>();
        int64_t num_indices = bndry_cpu.numel();

        int fd = open(file_paths[i].c_str(), O_RDONLY | O_DIRECT);
        AT_ASSERTM(fd >= 0, "GDS Gather: Failed to open boundary partition file");

        CUfileDescr_t descr;
        memset(&descr, 0, sizeof(descr));
        descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        descr.handle.fd = fd;

        CUfileHandle_t cf_handle;
        status = cuFileHandleRegister(&cf_handle, &descr);
        AT_ASSERTM(status.err == CU_FILE_SUCCESS, "GDS Gather: cuFileHandleRegister failed for boundary file");

        // Extract exact contiguous chunk runs
        int64_t idx = 0;
        while (idx < num_indices) {
          int64_t start = idx;
          int64_t end = idx + 1;

          while (end < num_indices && idx_ptr[end] == idx_ptr[end - 1] + 1) {
            end++;
          }

          int64_t chunk_len = end - start;
          size_t read_bytes = static_cast<size_t>(chunk_len * row_bytes);
          off_t file_offset_bytes = static_cast<off_t>(idx_ptr[start] * row_bytes);
          off_t buf_offset_bytes = static_cast<off_t>(offset * row_bytes);
          ssize_t bytes_read = 0;

          // async GDS direct read directly to GPU memory
          cuFileReadAsync(cf_handle, dst_data, &read_bytes, &file_offset_bytes, &buf_offset_bytes, &bytes_read, raw_stream);

          offset += chunk_len;
          idx = end;
        }

        // Wait on stream before closing boundary handle
        cudaStreamSynchronize(raw_stream);
        cuFileHandleDeregister(cf_handle);
        close(fd);
      }

      AT_ASSERTM(offset == dst.size(0),
                 "Gather GDS: copied size mismatch with destination");

      // Final synchronization for target file handle cleanup
      cudaStreamSynchronize(raw_stream);
      cuFileHandleDeregister(cf_target);
      close(fd_target);
    });
  });
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
