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
#include <cstring>

void gather_partitions_direct(
    int pid,
    std::vector<std::string> file_paths,
    torch::Tensor dst,
    std::vector<torch::Tensor> boundaries
) {
  AT_ASSERTM(dst.is_cuda(), "Destination must be a CUDA tensor");
  AT_ASSERTM(dst.is_contiguous(), "Destination must be contiguous");

  auto stream_obj = at::cuda::getCurrentCUDAStream(dst.get_device());
  cudaStream_t stream = stream_obj.stream();

  AT_ASSERTM(stream_obj != at::cuda::getDefaultCUDAStream(dst.get_device()),
             "Async gather requires a non-default CUDA stream");

  getH2DPool().run([=] {
    c10::cuda::CUDAStreamGuard guard(stream_obj);

    int64_t total_rows = dst.size(0);
    int64_t feat_dim = dst.numel() / total_rows;
    int64_t row_bytes = feat_dim * dst.element_size();
    uint8_t* dst_raw = reinterpret_cast<uint8_t*>(dst.data_ptr());

    size_t num_parts = file_paths.size();
    std::vector<int64_t> part_offsets(num_parts, 0);

    std::vector<int> open_fds;
    std::vector<CUfileHandle_t> registered_handles;

    auto cleanup_resources = [&]() {
      for (auto handle : registered_handles) {
        cuFileHandleDeregister(handle);
      }
      for (int fd : open_fds) {
        close(fd);
      }
    };

    // Calculate file offsets
    int target_fd = open(file_paths[pid].c_str(), O_RDONLY | O_DIRECT);
    AT_ASSERTM(target_fd >= 0, "Failed to open target partition file");

    off_t target_file_size = lseek(target_fd, 0, SEEK_END);
    int64_t target_nodes = target_file_size / row_bytes;

    int64_t curr_offset = target_nodes;
    size_t total_ops = 1; // 1 operation for target partition

    for (size_t i = 0; i < num_parts; ++i) {
      if (static_cast<int>(i) != pid && boundaries[i].defined() && boundaries[i].numel() > 0) {
        part_offsets[i] = curr_offset;
        curr_offset += boundaries[i].numel();
        total_ops += boundaries[i].numel(); // Upper-bound on chunk count for allocation
      }
    }

    AT_ASSERTM(curr_offset == total_rows, "Gather offset mismatch with destination size");

    // Pre-allocate argument buffers to guarantee pointer stability across async reads
    std::vector<size_t> size_args;
    std::vector<off_t> file_off_args;
    std::vector<off_t> dev_off_args;
    std::vector<ssize_t> bytes_read_args;

    size_args.reserve(total_ops);
    file_off_args.reserve(total_ops);
    dev_off_args.reserve(total_ops);
    bytes_read_args.reserve(total_ops);

    // Register stream with cuFile to prevent internal state collision
    CUfileError_t stream_reg_status = cuFileStreamRegister(stream, 0);
    AT_ASSERTM(stream_reg_status.err == CU_FILE_SUCCESS, "cuFileStreamRegister failed");

    // 1. Enqueue Target Partition Read (Async)
    CUfileDescr_t target_desc;
    memset(&target_desc, 0, sizeof(CUfileDescr_t));
    target_desc.handle.fd = target_fd;
    target_desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    CUfileHandle_t target_handle;
    CUfileError_t target_status = cuFileHandleRegister(&target_handle, &target_desc);
    if (target_status.err != CU_FILE_SUCCESS) {
      close(target_fd);
      cuFileStreamDeregister(stream);
      AT_ERROR("cuFileHandleRegister failed for target partition");
    }

    open_fds.push_back(target_fd);
    registered_handles.push_back(target_handle);

    size_args.push_back(static_cast<size_t>(target_file_size));
    file_off_args.push_back(0);
    dev_off_args.push_back(0);
    bytes_read_args.push_back(0);

    cuFileReadAsync(
        target_handle,
        dst_raw,
        &size_args.back(),
        &file_off_args.back(),
        &dev_off_args.back(),
        &bytes_read_args.back(),
        stream
    );

    // 2. Enqueue Boundary Partition Reads (Async)
    std::vector<torch::Tensor> host_boundaries(num_parts);

    for (size_t i = 0; i < num_parts; ++i) {
      if (static_cast<int>(i) == pid || !boundaries[i].defined() || boundaries[i].numel() == 0) {
        continue;
      }

      int fd = open(file_paths[i].c_str(), O_RDONLY | O_DIRECT);
      if (fd < 0) continue;

      CUfileDescr_t desc;
      memset(&desc, 0, sizeof(CUfileDescr_t));
      desc.handle.fd = fd;
      desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

      CUfileHandle_t handle;
      CUfileError_t reg_status = cuFileHandleRegister(&handle, &desc);
      if (reg_status.err != CU_FILE_SUCCESS) {
        close(fd);
        continue;
      }

      open_fds.push_back(fd);
      registered_handles.push_back(handle);

      host_boundaries[i] = boundaries[i].to(at::kCPU).contiguous();
      const int64_t* idx_ptr = host_boundaries[i].data_ptr<int64_t>();
      int64_t num_boundary_nodes = host_boundaries[i].numel();
      int64_t start_row_offset = part_offsets[i];

      int64_t idx = 0;
      while (idx < num_boundary_nodes) {
        int64_t range_start_node = idx_ptr[idx];
        int64_t range_len = 1;

        while (idx + range_len < num_boundary_nodes && 
               idx_ptr[idx + range_len] == range_start_node + range_len) {
          range_len++;
        }

        off_t file_offset = range_start_node * row_bytes;
        size_t read_bytes = range_len * row_bytes;
        uint8_t* dst_ptr = dst_raw + ((start_row_offset + idx) * row_bytes);

        size_args.push_back(read_bytes);
        file_off_args.push_back(file_offset);
        dev_off_args.push_back(0);
        bytes_read_args.push_back(0);

        cuFileReadAsync(
            handle,
            dst_ptr,
            &size_args.back(),
            &file_off_args.back(),
            &dev_off_args.back(),
            &bytes_read_args.back(),
            stream
        );

        idx += range_len;
      }
    }

    // 3. Synchronize stream and unregister cuFile stream state
    cudaStreamSynchronize(stream);
    cuFileStreamDeregister(stream);

    // 4. Clean up handles and file descriptors
    cleanup_resources();
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
