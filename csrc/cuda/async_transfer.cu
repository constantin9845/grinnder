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

#include <cerrno>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

constexpr size_t ALIGN_SIZE = 512;


struct GDSAsyncParams {
    size_t read_bytes = 0;
    off_t file_offset = 0;
    off_t buf_offset = 0;
    ssize_t bytes_read = 0;
    bool is_direct_gds = false;
    std::string file_name;
};


// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------

inline bool is_aligned(uintptr_t val) {
    return (val % ALIGN_SIZE) == 0;
}


inline void check_cufile(
    CUfileError_t status,
    const char* operation
) {
    if (status.err != CU_FILE_SUCCESS) {
        std::cerr
            << "[GDS ERROR] "
            << operation
            << " failed"
            << " | error_code="
            << static_cast<int>(status.err)
            << std::endl;

        throw std::runtime_error(
            std::string("cuFile error in ") + operation
        );
    }
}


inline void check_cuda(
    cudaError_t status,
    const char* operation
) {
    if (status != cudaSuccess) {
        std::cerr
            << "[CUDA ERROR] "
            << operation
            << ": "
            << cudaGetErrorString(status)
            << std::endl;

        throw std::runtime_error(
            std::string("CUDA error in ") + operation
        );
    }
}


// ------------------------------------------------------------
// Main GDS gather
// ------------------------------------------------------------

void gather_partitions_direct(
    int pid,
    std::vector<std::string> file_paths,
    std::vector<int64_t> num_nodes,
    torch::Tensor dst,
    std::vector<torch::Tensor> boundaries
) {
    AT_ASSERTM(
        dst.is_cuda(),
        "Destination must be a CUDA tensor"
    );

    AT_ASSERTM(
        dst.is_contiguous(),
        "Destination must be contiguous"
    );

    AT_ASSERTM(
        static_cast<size_t>(pid) < file_paths.size(),
        "Invalid pid"
    );

    AT_ASSERTM(
        file_paths.size() == num_nodes.size(),
        "file_paths and num_nodes size mismatch"
    );

    AT_ASSERTM(
        boundaries.size() == file_paths.size(),
        "boundaries and file_paths size mismatch"
    );

    const int device_id = dst.get_device();

    c10::cuda::CUDAGuard device_guard(device_id);

    // ------------------------------------------------------------
    // Use the current CUDA stream.
    //
    // Python puts this operation inside:
    //
    //     with torch.cuda.stream(stream):
    //
    // so this should be the requested stream.
    // ------------------------------------------------------------

    auto stream =
        at::cuda::getCurrentCUDAStream(device_id);

    cudaStream_t raw_stream =
        stream.stream();


    // ------------------------------------------------------------
    // Tensor information
    // ------------------------------------------------------------

    uint8_t* dst_data =
        reinterpret_cast<uint8_t*>(dst.data_ptr());

    const int64_t total_rows =
        dst.size(0);

    const int64_t feat_dim =
        dst.numel() / total_rows;

    const int64_t element_size =
        static_cast<int64_t>(dst.element_size());

    const int64_t row_bytes =
        feat_dim * element_size;

    const size_t total_tensor_bytes =
        static_cast<size_t>(dst.numel()) *
        static_cast<size_t>(dst.element_size());


    std::cout
        << "[GDS]"
        << " pid=" << pid
        << " device=" << device_id
        << " rows=" << total_rows
        << " feature_dim=" << feat_dim
        << " element_size=" << element_size
        << " row_bytes=" << row_bytes
        << " total_bytes=" << total_tensor_bytes
        << std::endl;


    // ------------------------------------------------------------
    // Register entire GPU destination buffer.
    // ------------------------------------------------------------

    CUfileError_t buf_status =
        cuFileBufRegister(
            dst_data,
            total_tensor_bytes,
            0
        );

    check_cufile(
        buf_status,
        "cuFileBufRegister"
    );

    std::cout
        << "[GDS] GPU buffer registered"
        << " ptr=" << static_cast<void*>(dst_data)
        << " bytes=" << total_tensor_bytes
        << std::endl;


    // ------------------------------------------------------------
    // Keep all async state alive until the final synchronize.
    // ------------------------------------------------------------

    std::vector<std::unique_ptr<GDSAsyncParams>>
        async_params_keeper;

    std::vector<void*>
        deferred_free_buffers;


    // ------------------------------------------------------------
    // Read helper
    // ------------------------------------------------------------

    auto perform_read =
        [&](
            CUfileHandle_t cf_handle,
            int raw_fd,
            off_t raw_file_offset,
            size_t raw_bytes,
            off_t raw_buf_offset,
            const std::string& path_tag
        ) {

        AT_ASSERTM(
            raw_bytes > 0,
            "GDS read size must be > 0"
        );


        uintptr_t target_gpu_ptr =
            reinterpret_cast<uintptr_t>(
                dst_data + raw_buf_offset
            );


        const bool file_aligned =
            is_aligned(
                static_cast<uintptr_t>(
                    raw_file_offset
                )
            );

        const bool size_aligned =
            is_aligned(raw_bytes);

        const bool gpu_aligned =
            is_aligned(target_gpu_ptr);


        // ========================================================
        // DIRECT GDS
        // ========================================================

        if (
            file_aligned &&
            size_aligned &&
            gpu_aligned
        ) {

            auto p =
                std::make_unique<GDSAsyncParams>();

            p->read_bytes =
                raw_bytes;

            p->file_offset =
                raw_file_offset;

            p->buf_offset =
                raw_buf_offset;

            p->bytes_read =
                0;

            p->is_direct_gds =
                true;

            p->file_name =
                path_tag;


            // IMPORTANT:
            //
            // cuFileReadAsync receives:
            //
            //   base registered GPU pointer
            //
            // and a device pointer offset.
            //
            // The GPU buffer itself was registered above.

            off_t dev_ptr_offset =
                raw_buf_offset;


            CUfileError_t status =
                cuFileReadAsync(
                    cf_handle,
                    dst_data,
                    &p->read_bytes,
                    &p->file_offset,
                    &dev_ptr_offset,
                    &p->bytes_read,
                    raw_stream
                );


            if (status.err != CU_FILE_SUCCESS) {

                std::cerr
                    << "\n[GDS ERROR] cuFileReadAsync failed"
                    << "\n  file="
                    << path_tag
                    << "\n  file_offset="
                    << raw_file_offset
                    << "\n  gpu_offset="
                    << raw_buf_offset
                    << "\n  bytes="
                    << raw_bytes
                    << "\n  gpu_ptr="
                    << static_cast<void*>(
                        dst_data + raw_buf_offset
                    )
                    << "\n  error_code="
                    << static_cast<int>(status.err)
                    << std::endl;

                throw std::runtime_error(
                    "cuFileReadAsync failed"
                );
            }


            std::cout
                << "[GDS SUBMIT]"
                << " file=" << path_tag
                << " file_off=" << raw_file_offset
                << " gpu_off=" << raw_buf_offset
                << " bytes=" << raw_bytes
                << std::endl;


            async_params_keeper.push_back(
                std::move(p)
            );

        }

        // ========================================================
        // ALIGNED CPU FALLBACK
        //
        // This exists only for cases where the file offset,
        // size, or GPU address isn't 512-byte aligned.
        // ========================================================

        else {

            std::cerr
                << "[GDS FALLBACK]"
                << " file=" << path_tag
                << " file_offset=" << raw_file_offset
                << " bytes=" << raw_bytes
                << " gpu_offset=" << raw_buf_offset
                << " gpu_aligned=" << gpu_aligned
                << " file_aligned=" << file_aligned
                << " size_aligned=" << size_aligned
                << std::endl;


            const off_t aligned_file_off =
                (
                    raw_file_offset /
                    static_cast<off_t>(ALIGN_SIZE)
                ) *
                static_cast<off_t>(ALIGN_SIZE);


            const size_t file_head_pad =
                static_cast<size_t>(
                    raw_file_offset -
                    aligned_file_off
                );


            const size_t aligned_read_bytes =
                (
                    file_head_pad +
                    raw_bytes +
                    ALIGN_SIZE -
                    1
                ) /
                ALIGN_SIZE *
                ALIGN_SIZE;


            void* cpu_aligned_buf =
                nullptr;


            int ret =
                posix_memalign(
                    &cpu_aligned_buf,
                    ALIGN_SIZE,
                    aligned_read_bytes
                );


            AT_ASSERTM(
                ret == 0,
                "Failed to allocate aligned bounce buffer"
            );


            ssize_t bytes_read =
                pread(
                    raw_fd,
                    cpu_aligned_buf,
                    aligned_read_bytes,
                    aligned_file_off
                );


            if (
                bytes_read <
                static_cast<ssize_t>(
                    file_head_pad +
                    raw_bytes
                )
            ) {

                free(cpu_aligned_buf);

                throw std::runtime_error(
                    "GDS fallback pread failed"
                );
            }


            uint8_t* src_payload =
                reinterpret_cast<uint8_t*>(
                    cpu_aligned_buf
                ) +
                file_head_pad;


            cudaError_t cuda_status =
                cudaMemcpyAsync(
                    dst_data + raw_buf_offset,
                    src_payload,
                    raw_bytes,
                    cudaMemcpyHostToDevice,
                    raw_stream
                );


            check_cuda(
                cuda_status,
                "cudaMemcpyAsync"
            );


            auto p =
                std::make_unique<GDSAsyncParams>();

            p->read_bytes =
                raw_bytes;

            p->file_offset =
                raw_file_offset;

            p->buf_offset =
                raw_buf_offset;

            p->bytes_read =
                static_cast<ssize_t>(
                    raw_bytes
                );

            p->is_direct_gds =
                false;

            p->file_name =
                path_tag;


            async_params_keeper.push_back(
                std::move(p)
            );


            // Must remain alive until the CUDA copy completes.
            deferred_free_buffers.push_back(
                cpu_aligned_buf
            );
        }
    };


    // ============================================================
    // 1. TARGET PARTITION
    // ============================================================

    int64_t offset =
        num_nodes[pid];


    const size_t target_bytes =
        static_cast<size_t>(
            num_nodes[pid] *
            row_bytes
        );


    int fd_target =
        open(
            file_paths[pid].c_str(),
            O_RDONLY | O_DIRECT
        );


    if (fd_target < 0) {

        std::cerr
            << "[GDS ERROR] Failed to open target"
            << " file=" << file_paths[pid]
            << " errno=" << errno
            << " (" << std::strerror(errno) << ")"
            << std::endl;

        throw std::runtime_error(
            "Failed to open target partition"
        );
    }


    CUfileDescr_t descr_target;

    std::memset(
        &descr_target,
        0,
        sizeof(descr_target)
    );


    descr_target.type =
        CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    descr_target.handle.fd =
        fd_target;


    CUfileHandle_t cf_target =
        nullptr;


    CUfileError_t status =
        cuFileHandleRegister(
            &cf_target,
            &descr_target
        );


    check_cufile(
        status,
        "cuFileHandleRegister(target)"
    );


    std::cout
        << "[GDS] Target partition"
        << " pid=" << pid
        << " rows=" << num_nodes[pid]
        << " bytes=" << target_bytes
        << std::endl;


    if (target_bytes > 0) {

        perform_read(
            cf_target,
            fd_target,
            0,
            target_bytes,
            0,
            file_paths[pid]
        );
    }


    // ============================================================
    // 2. BOUNDARY PARTITIONS
    //
    // NO contiguous grouping.
    //
    // Every boundary index becomes one async read.
    //
    // We intentionally DO NOT synchronize here.
    // ============================================================

    for (
        size_t i = 0;
        i < file_paths.size();
        ++i
    ) {

        if (static_cast<int>(i) == pid) {
            continue;
        }


        torch::Tensor bndry =
            boundaries[i];


        if (!bndry.defined()) {
            continue;
        }


        if (bndry.numel() == 0) {
            continue;
        }


        // --------------------------------------------------------
        // Make indices CPU contiguous.
        // --------------------------------------------------------

        torch::Tensor bndry_cpu =
            bndry
                .to(torch::kCPU, torch::kLong)
                .contiguous();


        const int64_t* idx_ptr =
            bndry_cpu.data_ptr<int64_t>();


        const int64_t num_indices =
            bndry_cpu.numel();


        // --------------------------------------------------------
        // Open boundary partition once.
        // --------------------------------------------------------

        int fd =
            open(
                file_paths[i].c_str(),
                O_RDONLY | O_DIRECT
            );


        if (fd < 0) {

            std::cerr
                << "[GDS ERROR] Failed to open boundary"
                << " pid=" << i
                << " file=" << file_paths[i]
                << " errno=" << errno
                << " (" << std::strerror(errno) << ")"
                << std::endl;

            throw std::runtime_error(
                "Failed to open boundary partition"
            );
        }


        CUfileDescr_t descr;

        std::memset(
            &descr,
            0,
            sizeof(descr)
        );


        descr.type =
            CU_FILE_HANDLE_TYPE_OPAQUE_FD;

        descr.handle.fd =
            fd;


        CUfileHandle_t cf_handle =
            nullptr;


        status =
            cuFileHandleRegister(
                &cf_handle,
                &descr
            );


        check_cufile(
            status,
            "cuFileHandleRegister(boundary)"
        );


        std::cout
            << "[GDS] Boundary"
            << " pid=" << i
            << " rows=" << num_indices
            << std::endl;


        // --------------------------------------------------------
        // One read per boundary row.
        //
        // No grouping.
        // --------------------------------------------------------

        for (
            int64_t k = 0;
            k < num_indices;
            ++k
        ) {

            const int64_t node_idx =
                idx_ptr[k];


            if (
                node_idx < 0 ||
                node_idx >= num_nodes[i]
            ) {

                cuFileHandleDeregister(
                    cf_handle
                );

                close(fd);

                throw std::runtime_error(
                    "Boundary index out of range"
                );
            }


            const size_t read_bytes =
                static_cast<size_t>(
                    row_bytes
                );


            const off_t file_offset_bytes =
                static_cast<off_t>(
                    node_idx *
                    row_bytes
                );


            const off_t buf_offset_bytes =
                static_cast<off_t>(
                    offset *
                    row_bytes
                );


            perform_read(
                cf_handle,
                fd,
                file_offset_bytes,
                read_bytes,
                buf_offset_bytes,
                file_paths[i]
            );


            ++offset;
        }


        // --------------------------------------------------------
        // IMPORTANT:
        //
        // DO NOT synchronize here.
        //
        // The reads from this partition remain queued while
        // subsequent partitions are submitted.
        // --------------------------------------------------------

        cuFileHandleDeregister(
            cf_handle
        );

        close(fd);


        std::cout
            << "[GDS] Submitted boundary"
            << " pid=" << i
            << " rows=" << num_indices
            << std::endl;
    }


    // ============================================================
    // 3. Verify destination size before waiting.
    // ============================================================

    AT_ASSERTM(
        offset == total_rows,
        "Gather GDS: copied size mismatch"
    );


    std::cout
        << "[GDS] All reads submitted"
        << " pid=" << pid
        << " total_rows=" << offset
        << std::endl;


    // ============================================================
    // 4. ONE FINAL SYNCHRONIZATION
    // ============================================================

    cudaError_t sync_status =
        cudaStreamSynchronize(
            raw_stream
        );


    check_cuda(
        sync_status,
        "cudaStreamSynchronize"
    );


    std::cout
        << "[GDS] Stream synchronized"
        << " pid=" << pid
        << std::endl;


    // ============================================================
    // 5. Verify async read results
    // ============================================================

    size_t total_gds_bytes = 0;
    size_t total_fallback_bytes = 0;


    for (
        const auto& item :
        async_params_keeper
    ) {

        if (item->is_direct_gds) {

            std::cout
                << "[GDS COMPLETE]"
                << " file=" << item->file_name
                << " requested=" << item->read_bytes
                << " completed=" << item->bytes_read
                << std::endl;


            if (
                item->bytes_read !=
                static_cast<ssize_t>(
                    item->read_bytes
                )
            ) {

                std::cerr
                    << "[GDS ERROR]"
                    << " Read byte mismatch"
                    << " file=" << item->file_name
                    << " requested="
                    << item->read_bytes
                    << " completed="
                    << item->bytes_read
                    << std::endl;

                throw std::runtime_error(
                    "GDS read byte count mismatch"
                );
            }


            total_gds_bytes +=
                static_cast<size_t>(
                    item->bytes_read
                );

        } else {

            total_fallback_bytes +=
                static_cast<size_t>(
                    item->bytes_read
                );
        }
    }


    std::cout
        << "[GDS Verification]"
        << " Pid=" << pid
        << " Direct GDS="
        << total_gds_bytes
        << " bytes"
        << " Fallback="
        << total_fallback_bytes
        << " bytes"
        << std::endl;


    // ============================================================
    // 6. Free fallback buffers AFTER CUDA synchronization.
    // ============================================================

    for (
        void* buf :
        deferred_free_buffers
    ) {
        free(buf);
    }

    deferred_free_buffers.clear();


    // ============================================================
    // 7. Cleanup target handle
    // ============================================================

    cuFileHandleDeregister(
        cf_target
    );

    close(fd_target);


    // ============================================================
    // 8. Deregister GPU buffer
    // ============================================================

    cuFileError_t dummy;

    dummy =
        cuFileBufDeregister(
            dst_data
        );

    if (dummy.err != CU_FILE_SUCCESS) {

        std::cerr
            << "[GDS WARNING]"
            << " cuFileBufDeregister failed"
            << " error_code="
            << static_cast<int>(
                dummy.err
            )
            << std::endl;
    }


    std::cout
        << "[GDS DONE]"
        << " pid=" << pid
        << " direct_bytes="
        << total_gds_bytes
        << " fallback_bytes="
        << total_fallback_bytes
        << std::endl;
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
