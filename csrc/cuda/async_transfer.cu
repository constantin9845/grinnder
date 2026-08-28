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
#include <string>
#include <vector>

constexpr size_t ALIGN_SIZE = 512;


// ============================================================
// Async bookkeeping
// ============================================================

struct GDSAsyncParams {
    size_t read_bytes = 0;
    off_t file_offset = 0;
    off_t buf_offset = 0;

    // cuFileReadAsync writes the completion result here.
    ssize_t bytes_read = -1;

    bool is_direct_gds = false;
    std::string file_name;
};


// ============================================================
// Helpers
// ============================================================

inline bool is_aligned(uintptr_t value) {
    return (value % ALIGN_SIZE) == 0;
}


// Convert cuFile error into something printable.
//
// cuFile does NOT provide cuFileGetErrorString().
// CUFILE_ERRSTR() is the supported cuFile error-string macro.
//
// For CUDA-specific errors, use cuGetErrorString().
//
static std::string cufile_error_string(const CUfileError_t& status) {

    std::string result;

    result += "cuFile err=";
    result += std::to_string(static_cast<int>(status.err));

    if (status.err == CU_FILE_CUDA_DRIVER_ERROR) {

        const char* cuda_error_string = nullptr;

        CUresult r =
            cuGetErrorString(status.cu_err, &cuda_error_string);

        if (r == CUDA_SUCCESS && cuda_error_string != nullptr) {
            result +=
                " CUDA error=" +
                std::string(cuda_error_string);
        } else {
            result +=
                " CUDA error code=" +
                std::to_string(
                    static_cast<int>(status.cu_err)
                );
        }

    } else {

        const char* cufile_string =
            CUFILE_ERRSTR(status.err);

        if (cufile_string != nullptr) {
            result +=
                " (" +
                std::string(cufile_string) +
                ")";
        }
    }

    return result;
}


static void check_cufile(
    const CUfileError_t& status,
    const char* operation
) {
    if (status.err != CU_FILE_SUCCESS) {

        std::string message =
            std::string(operation) +
            " failed: " +
            cufile_error_string(status);

        throw std::runtime_error(message);
    }
}


static void check_cuda(
    cudaError_t status,
    const char* operation
) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) +
            " failed: " +
            cudaGetErrorString(status)
        );
    }
}


// ============================================================
// Main GDS gather
// ============================================================

void gather_partitions_direct(
    int pid,
    std::vector<std::string> file_paths,
    std::vector<int64_t> num_nodes,
    torch::Tensor dst,
    std::vector<torch::Tensor> boundaries
) {

    TORCH_CHECK(
        dst.is_cuda(),
        "GDS gather: destination must be CUDA"
    );

    TORCH_CHECK(
        dst.is_contiguous(),
        "GDS gather: destination must be contiguous"
    );

    TORCH_CHECK(
        static_cast<size_t>(pid) < file_paths.size(),
        "GDS gather: invalid pid"
    );

    TORCH_CHECK(
        file_paths.size() == num_nodes.size(),
        "GDS gather: file_paths and num_nodes size mismatch"
    );

    TORCH_CHECK(
        file_paths.size() == boundaries.size(),
        "GDS gather: file_paths and boundaries size mismatch"
    );


    const int device_id = dst.get_device();


    // ========================================================
    // Run on the existing H2D pool
    // ========================================================

    AT_DISPATCH_ALL_TYPES_AND(
        at::ScalarType::Half,
        dst.scalar_type(),
        "gather_partitions_gds",
        [&] {

            getH2DPool().run([=] {

                c10::cuda::CUDAGuard device_guard(device_id);

                auto stream =
                    at::cuda::getCurrentCUDAStream(device_id);

                cudaStream_t raw_stream =
                    stream.stream();


                // ====================================================
                // Tensor geometry
                // ====================================================

                uint8_t* dst_data =
                    reinterpret_cast<uint8_t*>(
                        dst.data_ptr<scalar_t>()
                    );

                TORCH_CHECK(
                    dst.dim() >= 2,
                    "GDS gather expects dst to have at least 2 dimensions"
                );

                const int64_t total_rows =
                    dst.size(0);

                const int64_t feature_dim =
                    dst.numel() / total_rows;

                const size_t row_bytes =
                    static_cast<size_t>(feature_dim) *
                    sizeof(scalar_t);

                const size_t total_tensor_bytes =
                    static_cast<size_t>(dst.numel()) *
                    sizeof(scalar_t);


                std::cout
                    << "[GDS] ==================================================\n"
                    << "[GDS] pid=" << pid << "\n"
                    << "[GDS] device=" << device_id << "\n"
                    << "[GDS] rows=" << total_rows << "\n"
                    << "[GDS] feature_dim=" << feature_dim << "\n"
                    << "[GDS] row_bytes=" << row_bytes << "\n"
                    << "[GDS] total_bytes=" << total_tensor_bytes << "\n";


                // ====================================================
                // Register entire destination GPU buffer
                // ====================================================

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
                    << "[GDS] GPU buffer registered: "
                    << static_cast<void*>(dst_data)
                    << " size="
                    << total_tensor_bytes
                    << "\n";


                // ====================================================
                // Keep all async parameter objects alive until
                // stream synchronization.
                // ====================================================

                std::vector<
                    std::unique_ptr<GDSAsyncParams>
                > async_params_keeper;

                async_params_keeper.reserve(
                    1024
                );


                // ====================================================
                // Fallback buffers
                // ====================================================

                std::vector<void*> deferred_free_buffers;


                // ====================================================
                // perform_read
                //
                // No contiguous grouping.
                //
                // One invocation = one IO operation.
                // ====================================================

                auto perform_read =
                    [&](
                        CUfileHandle_t cf_handle,
                        int raw_fd,
                        off_t raw_file_offset,
                        size_t raw_bytes,
                        off_t raw_buf_offset,
                        const std::string& path_tag
                    ) {

                        const uintptr_t gpu_address =
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
                            is_aligned(gpu_address);


                        // =================================================
                        // DIRECT GDS
                        // =================================================

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
                                -1;

                            p->is_direct_gds =
                                true;

                            p->file_name =
                                path_tag;


                            // IMPORTANT:
                            //
                            // devPtr_base MUST be dst_data because
                            // dst_data is what we registered.
                            //
                            // devPtr_offset tells GDS where inside
                            // dst_data to put the data.
                            //
                            off_t dev_ptr_offset =
                                raw_buf_offset;


                            CUfileError_t status =
                                cuFileReadAsync(
                                    cf_handle,

                                    // Registered base pointer.
                                    dst_data,

                                    &p->read_bytes,

                                    &p->file_offset,

                                    &dev_ptr_offset,

                                    &p->bytes_read,

                                    raw_stream
                                );


                            if (
                                status.err !=
                                CU_FILE_SUCCESS
                            ) {

                                std::cerr
                                    << "\n[GDS ERROR]\n"
                                    << "  file="
                                    << path_tag
                                    << "\n"
                                    << "  file_offset="
                                    << raw_file_offset
                                    << "\n"
                                    << "  buf_offset="
                                    << raw_buf_offset
                                    << "\n"
                                    << "  bytes="
                                    << raw_bytes
                                    << "\n"
                                    << "  gpu_address="
                                    << reinterpret_cast<void*>(
                                        gpu_address
                                    )
                                    << "\n"
                                    << "  error="
                                    << cufile_error_string(
                                        status
                                    )
                                    << "\n";

                                throw std::runtime_error(
                                    "cuFileReadAsync failed"
                                );
                            }


                            std::cout
                                << "[GDS SUBMIT] "
                                << path_tag
                                << " file_off="
                                << raw_file_offset
                                << " buf_off="
                                << raw_buf_offset
                                << " bytes="
                                << raw_bytes
                                << "\n";


                            async_params_keeper.push_back(
                                std::move(p)
                            );

                            return;
                        }


                        // =================================================
                        // FALLBACK
                        //
                        // File or GPU destination isn't aligned.
                        // Read through an aligned CPU buffer.
                        // =================================================

                        std::cerr
                            << "[GDS FALLBACK] "
                            << path_tag
                            << "\n"
                            << "  file_offset="
                            << raw_file_offset
                            << "\n"
                            << "  bytes="
                            << raw_bytes
                            << "\n"
                            << "  gpu_aligned="
                            << gpu_aligned
                            << "\n"
                            << "  file_aligned="
                            << file_aligned
                            << "\n"
                            << "  size_aligned="
                            << size_aligned
                            << "\n";


                        const off_t aligned_file_offset =
                            (
                                raw_file_offset /
                                ALIGN_SIZE
                            ) * ALIGN_SIZE;


                        const size_t file_head_pad =
                            static_cast<size_t>(
                                raw_file_offset -
                                aligned_file_offset
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


                        const int ret =
                            posix_memalign(
                                &cpu_aligned_buf,
                                ALIGN_SIZE,
                                aligned_read_bytes
                            );


                        TORCH_CHECK(
                            ret == 0,
                            "GDS fallback: "
                            "posix_memalign failed"
                        );


                        const ssize_t bytes_read =
                            pread(
                                raw_fd,
                                cpu_aligned_buf,
                                aligned_read_bytes,
                                aligned_file_offset
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
                                "GDS fallback: pread failed/incomplete"
                            );
                        }


                        uint8_t* src_payload =
                            reinterpret_cast<uint8_t*>(
                                cpu_aligned_buf
                            ) +
                            file_head_pad;


                        cudaError_t cuda_status =
                            cudaMemcpyAsync(
                                dst_data +
                                raw_buf_offset,

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
                            raw_bytes;

                        p->is_direct_gds =
                            false;

                        p->file_name =
                            path_tag;


                        async_params_keeper.push_back(
                            std::move(p)
                        );

                        deferred_free_buffers.push_back(
                            cpu_aligned_buf
                        );
                    };


                // ========================================================
                // 1. TARGET PARTITION
                // ========================================================

                int64_t offset =
                    num_nodes[pid];


                TORCH_CHECK(
                    offset >= 0,
                    "Invalid target partition node count"
                );


                int fd_target =
                    open(
                        file_paths[pid].c_str(),
                        O_RDONLY | O_DIRECT
                    );


                TORCH_CHECK(
                    fd_target >= 0,
                    "Failed to open target partition: ",
                    file_paths[pid],
                    " errno=",
                    errno,
                    " (",
                    std::strerror(errno),
                    ")"
                );


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


                CUfileError_t target_status =
                    cuFileHandleRegister(
                        &cf_target,
                        &descr_target
                    );


                check_cufile(
                    target_status,
                    "cuFileHandleRegister(target)"
                );


                std::cout
                    << "[GDS] Target partition pid="
                    << pid
                    << " rows="
                    << num_nodes[pid]
                    << " bytes="
                    << (
                        static_cast<size_t>(
                            num_nodes[pid]
                        ) *
                        row_bytes
                    )
                    << "\n";


                // Full target partition.
                perform_read(
                    cf_target,
                    fd_target,

                    0,

                    static_cast<size_t>(
                        num_nodes[pid]
                    ) *
                    row_bytes,

                    0,

                    file_paths[pid]
                );


                // ========================================================
                // 2. BOUNDARY PARTITIONS
                //
                // NO contiguous grouping.
                //
                // Every requested boundary row becomes one async read.
                // ========================================================

                for (
                    size_t i = 0;
                    i < file_paths.size();
                    ++i
                ) {

                    if (
                        static_cast<int>(i) ==
                        pid
                    ) {
                        continue;
                    }


                    auto bndry =
                        boundaries[i];


                    if (
                        !bndry.defined() ||
                        bndry.numel() == 0
                    ) {
                        continue;
                    }


                    // ----------------------------------------------
                    // Move indices to CPU.
                    // ----------------------------------------------

                    auto bndry_cpu =
                        bndry.to(
                            torch::kCPU,
                            torch::kLong
                        ).contiguous();


                    const int64_t* idx_ptr =
                        bndry_cpu.data_ptr<int64_t>();


                    const int64_t num_indices =
                        bndry_cpu.numel();


                    std::cout
                        << "[GDS] Boundary pid="
                        << i
                        << " rows="
                        << num_indices
                        << "\n";


                    // ----------------------------------------------
                    // Open boundary file.
                    // ----------------------------------------------

                    int fd =
                        open(
                            file_paths[i].c_str(),
                            O_RDONLY | O_DIRECT
                        );


                    TORCH_CHECK(
                        fd >= 0,
                        "Failed to open boundary partition: ",
                        file_paths[i],
                        " errno=",
                        errno,
                        " (",
                        std::strerror(errno),
                        ")"
                    );


                    // ----------------------------------------------
                    // Register cuFile handle.
                    // ----------------------------------------------

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


                    CUfileError_t status =
                        cuFileHandleRegister(
                            &cf_handle,
                            &descr
                        );


                    if (
                        status.err !=
                        CU_FILE_SUCCESS
                    ) {

                        close(fd);

                        throw std::runtime_error(
                            "cuFileHandleRegister(boundary) failed: " +
                            cufile_error_string(status)
                        );
                    }


                    // ----------------------------------------------
                    // ONE READ PER BOUNDARY ROW.
                    // ----------------------------------------------

                    for (
                        int64_t k = 0;
                        k < num_indices;
                        ++k
                    ) {

                        const int64_t node_idx =
                            idx_ptr[k];


                        TORCH_CHECK(
                            node_idx >= 0 &&
                            node_idx < num_nodes[i],

                            "Boundary index out of range: "
                            "partition=",
                            i,
                            " node=",
                            node_idx,
                            " num_nodes=",
                            num_nodes[i]
                        );


                        const size_t read_bytes =
                            row_bytes;


                        const off_t file_offset_bytes =
                            static_cast<off_t>(
                                node_idx
                            ) *
                            static_cast<off_t>(
                                row_bytes
                            );


                        const off_t buf_offset_bytes =
                            static_cast<off_t>(
                                offset
                            ) *
                            static_cast<off_t>(
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


                        // Only print periodically so that
                        // 50k rows don't flood stdout.
                        if (
                            (k % 10000) == 0
                        ) {

                            std::cout
                                << "[GDS] Boundary pid="
                                << i
                                << " submitted "
                                << k
                                << "/"
                                << num_indices
                                << "\n";
                        }
                    }


                    std::cout
                        << "[GDS] Boundary pid="
                        << i
                        << " all "
                        << num_indices
                        << " reads submitted\n";


                    // IMPORTANT:
                    //
                    // Wait before deregistering the file handle.
                    //
                    // cuFileReadAsync is asynchronous.
                    // The handle must stay alive until the
                    // operations using it have completed.
                    //
                    cudaError_t sync_status =
                        cudaStreamSynchronize(
                            raw_stream
                        );


                    check_cuda(
                        sync_status,
                        "cudaStreamSynchronize(boundary)"
                    );


                    // ----------------------------------------------
                    // Verify completed reads for this file.
                    // ----------------------------------------------

                    size_t boundary_gds_bytes = 0;
                    size_t boundary_fallback_bytes = 0;


                    for (
                        const auto& item :
                        async_params_keeper
                    ) {

                        if (
                            item->file_name !=
                            file_paths[i]
                        ) {
                            continue;
                        }


                        if (
                            item->is_direct_gds
                        ) {

                            if (
                                item->bytes_read !=
                                static_cast<ssize_t>(
                                    item->read_bytes
                                )
                            ) {

                                throw std::runtime_error(
                                    "GDS read byte count mismatch for " +
                                    item->file_name +
                                    ": requested=" +
                                    std::to_string(
                                        item->read_bytes
                                    ) +
                                    " completed=" +
                                    std::to_string(
                                        item->bytes_read
                                    )
                                );
                            }


                            boundary_gds_bytes +=
                                static_cast<size_t>(
                                    item->bytes_read
                                );

                        } else {

                            boundary_fallback_bytes +=
                                static_cast<size_t>(
                                    item->bytes_read
                                );
                        }
                    }


                    std::cout
                        << "[GDS COMPLETE] Boundary pid="
                        << i
                        << " GDS="
                        << boundary_gds_bytes
                        << " bytes"
                        << " fallback="
                        << boundary_fallback_bytes
                        << " bytes"
                        << "\n";


                    // ----------------------------------------------
                    // Now it is safe to destroy the file handle.
                    // ----------------------------------------------

                    CUfileError_t dereg_status =
                        cuFileHandleDeregister(
                            cf_handle
                        );


                    if (
                        dereg_status.err !=
                        CU_FILE_SUCCESS
                    ) {

                        std::cerr
                            << "[GDS WARNING] "
                            << "cuFileHandleDeregister(boundary) "
                            << cufile_error_string(
                                dereg_status
                            )
                            << "\n";
                    }


                    close(fd);


                    // ----------------------------------------------
                    // Free fallback buffers belonging to work
                    // already synchronized above.
                    // ----------------------------------------------

                    for (
                        void* buf :
                        deferred_free_buffers
                    ) {
                        free(buf);
                    }

                    deferred_free_buffers.clear();
                }


                // ========================================================
                // 3. FINAL STREAM SYNCHRONIZATION
                // ========================================================

                cudaError_t final_sync =
                    cudaStreamSynchronize(
                        raw_stream
                    );


                check_cuda(
                    final_sync,
                    "Final cudaStreamSynchronize"
                );


                // ========================================================
                // VERIFY ALL ASYNC OPERATIONS
                // ========================================================

                size_t total_gds_bytes = 0;
                size_t total_fallback_bytes = 0;


                for (
                    const auto& item :
                    async_params_keeper
                ) {

                    if (
                        item->is_direct_gds
                    ) {

                        std::cout
                            << "[GDS COMPLETE] "
                            << item->file_name
                            << " requested="
                            << item->read_bytes
                            << " completed="
                            << item->bytes_read
                            << "\n";


                        TORCH_CHECK(
                            item->bytes_read ==
                            static_cast<ssize_t>(
                                item->read_bytes
                            ),

                            "GDS read byte count mismatch: "
                            "file=",
                            item->file_name,
                            " requested=",
                            item->read_bytes,
                            " completed=",
                            item->bytes_read
                        );


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


                // ========================================================
                // TARGET HANDLE CLEANUP
                // ========================================================

                CUfileError_t target_deregister_status =
                    cuFileHandleDeregister(
                        cf_target
                    );


                if (
                    target_deregister_status.err !=
                    CU_FILE_SUCCESS
                ) {

                    std::cerr
                        << "[GDS WARNING] "
                        << "target handle deregister failed: "
                        << cufile_error_string(
                            target_deregister_status
                        )
                        << "\n";
                }


                close(fd_target);


                // ========================================================
                // Free remaining fallback buffers.
                // ========================================================

                for (
                    void* buf :
                    deferred_free_buffers
                ) {
                    free(buf);
                }

                deferred_free_buffers.clear();


                // ========================================================
                // Deregister GPU buffer.
                // ========================================================

                CUfileError_t buffer_deregister_status =
                    cuFileBufDeregister(
                        dst_data
                    );


                if (
                    buffer_deregister_status.err !=
                    CU_FILE_SUCCESS
                ) {

                    std::cerr
                        << "[GDS WARNING] "
                        << "cuFileBufDeregister failed: "
                        << cufile_error_string(
                            buffer_deregister_status
                        )
                        << "\n";
                }


                // ========================================================
                // Final offset check
                // ========================================================

                TORCH_CHECK(
                    offset == total_rows,

                    "Gather GDS: copied size mismatch. "
                    "offset=",
                    offset,
                    " dst_rows=",
                    total_rows
                );


                // ========================================================
                // Final stats
                // ========================================================

                std::cout
                    << "[GDS Verification] "
                    << "Pid="
                    << pid
                    << " | Direct GDS="
                    << total_gds_bytes
                    << " bytes"
                    << " | Fallback="
                    << total_fallback_bytes
                    << " bytes"
                    << "\n";


                std::cout
                    << "[GDS] ==================================================\n";
            });
        }
    );
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
