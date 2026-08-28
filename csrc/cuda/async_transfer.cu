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
#include <cuda_runtime.h>
#include <cufile.h>

#include <fcntl.h>
#include <unistd.h>
#include <vector>
#include <string>
#include <memory>

void gather_partitions_direct(
    int pid,
    const std::vector<std::string>& file_paths,
    torch::Tensor dst,
    const std::vector<torch::Tensor>& boundaries) 
{
    AT_ASSERTM(dst.is_cuda(), "Destination must be a CUDA tensor");
    AT_ASSERTM(dst.is_contiguous(), "Destination must be contiguous");

    int device_id = dst.get_device();
    auto stream = at::cuda::getCurrentCUDAStream(device_id);

    // Capture tensor memory address and shape metadata
    uint8_t* dst_ptr = reinterpret_cast<uint8_t*>(dst.data_ptr());
    size_t total_dst_bytes = dst.numel() * dst.element_size();
    int64_t feat_dim = dst.numel() / dst.size(0);
    int64_t row_bytes = feat_dim * dst.element_size();
    size_t num_parts = file_paths.size();

    // 1. Register CUDA memory buffer for Direct GDS DMA transfers
    CUfileError_t buf_status = cuFileBufRegister(dst_ptr, total_dst_bytes, 0);
    AT_ASSERTM(buf_status.err == CU_FILE_SUCCESS, "cuFileBufRegister failed for dst tensor");

    // Launch execution worker on host thread pool
    getH2DPool().run([=]() {
        // Ensure CUDA context is bound to destination GPU in this background thread
        cudaSetDevice(device_id);

        int64_t current_row_offset = 0;

        // Container to keep file handles, descriptors, and async parameters alive until sync
        struct FileContext {
            int fd = -1;
            CUfileHandle_t handle = nullptr;
        };

        struct AsyncReadParams {
            size_t bytes_to_read;
            off_t file_offset;
            off_t dev_offset;
            ssize_t bytes_read;
        };

        std::vector<FileContext> active_files;
        std::vector<std::shared_ptr<AsyncReadParams>> param_store;

        // --------------------------------------------------------------------
        // Step A: Target Partition Read
        // --------------------------------------------------------------------
        std::string target_path = file_paths[pid];
        int fd = open(target_path.c_str(), O_RDONLY | O_DIRECT);
        AT_ASSERTM(fd >= 0, "Failed to open target partition file");

        CUfileDescr_t descr{};
        descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        descr.handle.fd = fd;

        CUfileHandle_t cf_handle;
        CUfileError_t status = cuFileHandleRegister(&cf_handle, &descr);
        AT_ASSERTM(status.err == CU_FILE_SUCCESS, "cuFileHandleRegister failed for target partition");

        active_files.push_back({fd, cf_handle});

        off_t file_size_off = lseek(fd, 0, SEEK_END);
        int64_t target_rows = file_size_off / row_bytes;

        auto target_params = std::make_shared<AsyncReadParams>();
        target_params->bytes_to_read = static_cast<size_t>(file_size_off);
        target_params->file_offset = 0;
        target_params->dev_offset = 0;
        target_params->bytes_read = 0;
        param_store.push_back(target_params);

        CUfileError_t read_status = cuFileReadAsync(
            cf_handle,
            dst_ptr,
            &target_params->bytes_to_read,
            &target_params->file_offset,
            &target_params->dev_offset,
            &target_params->bytes_read,
            stream.stream()
        );
        AT_ASSERTM(read_status.err == CU_FILE_SUCCESS, "cuFileReadAsync failed on target partition");

        current_row_offset += target_rows;

        // --------------------------------------------------------------------
        // Step B: Boundary Partition Reads
        // --------------------------------------------------------------------
        for (size_t i = 0; i < num_parts; ++i) {
            if (static_cast<int>(i) == pid) continue;

            auto bndry = boundaries[i];
            if (bndry.numel() == 0) continue;

            torch::Tensor bndry_cpu = bndry.to(torch::kCPU, torch::kLong).contiguous();
            const int64_t* indices = bndry_cpu.data_ptr<int64_t>();
            int64_t num_indices = bndry_cpu.size(0);

            int b_fd = open(file_paths[i].c_str(), O_RDONLY | O_DIRECT);
            AT_ASSERTM(b_fd >= 0, "Failed to open boundary partition file");

            CUfileDescr_t b_descr{};
            b_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
            b_descr.handle.fd = b_fd;

            CUfileHandle_t b_handle;
            status = cuFileHandleRegister(&b_handle, &b_descr);
            AT_ASSERTM(status.err == CU_FILE_SUCCESS, "cuFileHandleRegister failed on boundary partition");

            active_files.push_back({b_fd, b_handle});

            // Coalesce adjacent row indices into contiguous blocks
            int64_t block_start = indices[0];
            int64_t block_len = 1;

            std::vector<std::pair<int64_t, int64_t>> blocks;
            for (int64_t k = 1; k < num_indices; ++k) {
                if (indices[k] == block_start + block_len) {
                    block_len++;
                } else {
                    blocks.push_back({block_start, block_len});
                    block_start = indices[k];
                    block_len = 1;
                }
            }
            blocks.push_back({block_start, block_len});

            // Issue cuFileReadAsync with heap-allocated persistent parameter blocks
            for (const auto& block : blocks) {
                int64_t start_row = block.first;
                int64_t length = block.second;

                auto blk_params = std::make_shared<AsyncReadParams>();
                blk_params->bytes_to_read = length * row_bytes;
                blk_params->file_offset = start_row * row_bytes;
                blk_params->dev_offset = current_row_offset * row_bytes;
                blk_params->bytes_read = 0;
                param_store.push_back(blk_params);

                CUfileError_t block_status = cuFileReadAsync(
                    b_handle,
                    dst_ptr,
                    &blk_params->bytes_to_read,
                    &blk_params->file_offset,
                    &blk_params->dev_offset,
                    &blk_params->bytes_read,
                    stream.stream()
                );
                AT_ASSERTM(block_status.err == CU_FILE_SUCCESS, "cuFileReadAsync failed on boundary block read");

                current_row_offset += length;
            }
        }

        // Wait for all async CUDA stream reads to complete before freeing handles/pointers
        cudaStreamSynchronize(stream.stream());

        // Deregister handles and close descriptors
        for (auto& ctx : active_files) {
            cuFileHandleDeregister(ctx.handle);
            close(ctx.fd);
        }

        cuFileBufDeregister(dst_ptr);
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
