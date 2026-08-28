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



#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cufile.h>
#include <fcntl.h>
#include <unistd.h>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <iostream>

constexpr size_t ALIGN_SIZE = 512;

struct GDSAsyncParams {
  size_t read_bytes;
  off_t file_offset;
  off_t buf_offset;
  ssize_t bytes_read; // Written asynchronously by cuFileReadAsync
  bool is_direct_gds;
  std::string file_name;
};

inline bool is_aligned(uintptr_t val) {
  return (val % ALIGN_SIZE) == 0;
}

void gather_partitions_direct(
    int pid,
    std::vector<std::string> file_paths,
    std::vector<int64_t> num_nodes,
    torch::Tensor dst,
    std::vector<torch::Tensor> boundaries
) {
  AT_ASSERTM(dst.is_cuda(), "Destination must be a CUDA tensor");
  AT_ASSERTM(dst.is_contiguous(), "Destination must be contiguous");

  const int device_id = dst.get_device();

  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, dst.scalar_type(), "gather_partitions_gds", [&] {
    getH2DPool().run([=] {
      c10::cuda::CUDAGuard device_guard(device_id);
      auto stream = at::cuda::getCurrentCUDAStream(device_id);
      cudaStream_t raw_stream = stream.stream();

      auto dst_data = reinterpret_cast<uint8_t*>(dst.data_ptr<scalar_t>());
      int64_t feat_dim = dst.numel() / dst.size(0);
      int64_t row_bytes = feat_dim * sizeof(scalar_t);
      size_t total_tensor_bytes = dst.numel() * sizeof(scalar_t);

      // Register GPU buffer with GDS
      CUfileError_t buf_status = cuFileBufRegister(dst_data, total_tensor_bytes, 0);
      AT_ASSERTM(buf_status.err == CU_FILE_SUCCESS, "GDS: cuFileBufRegister failed");

      std::vector<std::unique_ptr<GDSAsyncParams>> async_params_keeper;
      std::vector<void*> deferred_free_buffers;

      auto perform_read = [&](CUfileHandle_t cf_handle, int raw_fd, off_t raw_file_offset, 
                              size_t raw_bytes, off_t raw_buf_offset, const std::string& path_tag) {
        
        uintptr_t target_gpu_ptr = reinterpret_cast<uintptr_t>(dst_data + raw_buf_offset);

        if (is_aligned(raw_file_offset) && is_aligned(raw_bytes) && is_aligned(target_gpu_ptr)) {
          // Direct GDS Path
          auto p = std::make_unique<GDSAsyncParams>();
          p->read_bytes = raw_bytes;
          p->file_offset = raw_file_offset;
          p->buf_offset = raw_buf_offset;
          p->bytes_read = -1; // Sentinel value to verify async completion
          p->is_direct_gds = true;
          p->file_name = path_tag;

          off_t zero_dev_offset = 0;
          cuFileReadAsync(
              cf_handle,
              dst_data + raw_buf_offset,
              &p->read_bytes,
              &p->file_offset,
              &zero_dev_offset,
              &p->bytes_read,
              raw_stream
          );
          async_params_keeper.push_back(std::move(p));
        } else {
          // Aligned Fallback Path
          off_t aligned_file_off = (raw_file_offset / ALIGN_SIZE) * ALIGN_SIZE;
          off_t file_head_pad = raw_file_offset - aligned_file_off;
          size_t aligned_read_bytes = ((file_head_pad + raw_bytes + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE;

          void* cpu_aligned_buf = nullptr;
          int ret = posix_memalign(&cpu_aligned_buf, ALIGN_SIZE, aligned_read_bytes);
          AT_ASSERTM(ret == 0, "GDS Gather: Failed to allocate aligned bounce buffer");

          ssize_t bytes_read = pread(raw_fd, cpu_aligned_buf, aligned_read_bytes, aligned_file_off);
          AT_ASSERTM(bytes_read >= static_cast<ssize_t>(file_head_pad + raw_bytes), 
                     "GDS Gather: Fallback pread failed or read incomplete");

          uint8_t* src_payload = reinterpret_cast<uint8_t*>(cpu_aligned_buf) + file_head_pad;
          cudaMemcpyAsync(dst_data + raw_buf_offset, src_payload, raw_bytes, cudaMemcpyHostToDevice, raw_stream);

          auto p = std::make_unique<GDSAsyncParams>();
          p->read_bytes = raw_bytes;
          p->bytes_read = raw_bytes;
          p->is_direct_gds = false;
          p->file_name = path_tag;
          async_params_keeper.push_back(std::move(p));

          deferred_free_buffers.push_back(cpu_aligned_buf);
        }
      };

      // 1. Read Target Partition
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

      perform_read(cf_target, fd_target, 0, static_cast<size_t>(offset * row_bytes), 0, file_paths[pid]);

      // 2. Read Boundary Partition Chunks
      for (size_t i = 0; i < file_paths.size(); i++) {
        if ((int)i == pid) continue;

        auto bndry = boundaries[i];
        if (bndry.numel() == 0) continue;

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

          perform_read(cf_handle, fd, file_offset_bytes, read_bytes, buf_offset_bytes, file_paths[i]);

          offset += chunk_len;
          idx = end;
        }

        cudaStreamSynchronize(raw_stream);
        cuFileHandleDeregister(cf_handle);
        close(fd);

        for (void* buf : deferred_free_buffers) {
          free(buf);
        }
        deferred_free_buffers.clear();
      }

      // Final Stream Sync & Cleanup Target Handles
      cudaStreamSynchronize(raw_stream);
      cuFileHandleDeregister(cf_target);
      close(fd_target);

      for (void* buf : deferred_free_buffers) {
        free(buf);
      }

      cuFileBufDeregister(dst_data);

      AT_ASSERTM(offset == dst.size(0), "Gather GDS: copied size mismatch with destination");

      // Verification: Check I/O stats after stream synchronization
      size_t total_gds_bytes = 0;
      size_t total_fallback_bytes = 0;

      for (const auto& item : async_params_keeper) {
        if (item->is_direct_gds) {
          AT_ASSERTM(item->bytes_read == static_cast<ssize_t>(item->read_bytes),
                     "GDS Verification Failure: Read bytes mismatch for direct GDS call");
          total_gds_bytes += item->bytes_read;
        } else {
          total_fallback_bytes += item->bytes_read;
        }
      }

      std::cout << "[GDS Verification] Pid: " << pid 
                << " | Direct GDS: " << total_gds_bytes << " bytes"
                << " | Fallback: " << total_fallback_bytes << " bytes" << std::endl;
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
