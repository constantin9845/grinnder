#pragma once

#include <torch/extension.h>
#include <cufile.h>

// D2H: GPU -> CPU host tensor (on current CUDA stream, via D2H thread pool)
void d2h_copy_async(torch::Tensor src, torch::Tensor dst);

// H2D: CPU host tensor -> GPU (on current CUDA stream, via H2D thread pool)
void h2d_copy_async(torch::Tensor src, torch::Tensor dst);

// Gather: multiple host partitions -> one GPU tensor (via H2D thread pool)
// Layout: [intra(srcs[pid]) | boundary_from_p0 | boundary_from_p1 | ...]
void gather_partitions(int pid, std::vector<torch::Tensor> srcs,
                       torch::Tensor dst,
                       std::vector<torch::Tensor> boundaries);

void gather_partitions_direct(
    int pid,
    const std::vector<int>& fds,
    torch::Tensor dst,
    std::vector<torch::Tensor> boundaries);

void gather_activations_direct(
    const std::string& filepath,
    torch::Tensor dst);



// Scatter: one GPU tensor -> multiple host partitions with accumulation
// Layout matches gather. Accumulates into dst partitions.
// Synchronizes the stream before accumulation.
void scatter_partitions(int pid, torch::Tensor src,
                        std::vector<torch::Tensor> dsts,
                        std::vector<torch::Tensor> boundaries);

std::vector<int> open_files(const std::vector<std::string>& file_paths);
void close_files(const std::vector<int>& fds);

// Thread pool synchronization
void h2d_synchronize();
void d2h_synchronize();

int64_t device_write(
    int64_t fd,
    torch::Tensor src,
    torch::Tensor ppa_list,
    int sector_size
);

bool device_read_and_verify(
    int64_t fd,
    torch::Tensor expected_src,  /* Ground truth tensor (on CUDA) */
    torch::Tensor ppa_list,      /* Tensor of uint64 PPAs */
    int sector_size
);


int64_t register_fd(int raw_fd);

void gather_partitions_direct_raw(
  int pid,
  CUfileHandle_t dev_handle,                                    
  torch::Tensor dst,
  std::vector<torch::Tensor> boundaries,
  std::vector<torch::Tensor> boundary_ppas,       
  std::vector<torch::Tensor> boundary_page_offs,  
  int64_t row_bytes
);

void device_write_wait(int64_t handle_id);
