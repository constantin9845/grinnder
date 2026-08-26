"""Device buffer: partition-wise GPU tensor management with double buffering.

Only pool_size (default 2) partition slots are allocated at any time.
After processing, partitions are freed via resize_(0).
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor

from grinnder.buffer.host import HostBuffer
from grinnder.stats import stat

def _load_ops():
    """Lazy-load C++ extension ops."""
    try:
        import grinnder._C as _C
        return _C
    except ImportError:
        return None


class DeviceBuffer:
    """Partition-wise tensor storage on GPU.

    Manages per-partition GPU tensors with lazy allocation:
    - allocate(pid): resize storage to hold partition data
    - release(pid): resize storage to 0 (free GPU memory)

    Only pool_size partitions should be live at any time for double buffering.

    Args:
        num_parts: Number of partitions.
        part_sizes: Number of nodes (rows) per partition. For gathered tensors,
                    this should be the EXPANDED size (intra + boundary nodes).
        embedding_dim: Feature dimension.
        device: CUDA device string.
    """

    def __init__(
        self,
        num_parts: int,
        part_sizes: List[int],
        embedding_dim: int,
        device: str = "cuda:0",
    ):
        self.num_parts = num_parts
        self.part_sizes = part_sizes
        self.embedding_dim = embedding_dim
        self.device = device

        self._ops = _load_ops()

        # Pre-allocate tensor objects with zero storage (lazy allocation)
        self._tensors: List[Tensor] = []
        for i in range(num_parts):
            t = torch.empty(
                part_sizes[i], embedding_dim, dtype=torch.float32, device=device
            )
            t.untyped_storage().resize_(0)  # Free memory, keep tensor object
            self._tensors.append(t)

    def __getitem__(self, pid: int) -> Tensor:
        """Get GPU tensor for partition pid. Must be allocated first."""
        return self._tensors[pid]

    def __len__(self) -> int:
        return self.num_parts

    def allocate(self, pid: int) -> Tensor:
        """Allocate GPU memory for partition pid.

        Returns the allocated tensor.
        """
        t = self._tensors[pid]
        nbytes = t.numel() * t.element_size()
        if t.untyped_storage().size() < nbytes:
            t.untyped_storage().resize_(nbytes)
        return t

    def release(self, pid: int) -> None:
        """Free GPU memory for partition pid (resize storage to 0)."""
        tensor = self._tensors[pid]
        tensor.untyped_storage().resize_(0)
        tensor.grad = None

    def is_allocated(self, pid: int) -> bool:
        """Check if partition pid has GPU memory allocated."""
        return self._tensors[pid].untyped_storage().size() > 0

    def allocated_pids(self) -> List[int]:
        """Return partitions that currently occupy GPU storage."""
        return [pid for pid in range(self.num_parts) if self.is_allocated(pid)]

    @property
    def resident_bytes(self) -> int:
        """Current GPU bytes occupied by resident partition storage."""
        return sum(
            t.untyped_storage().size()
            for t in self._tensors
            if t.untyped_storage().size() > 0
        )

    # ------------------------------------------------------------------
    # Gather shortcut: allocate + gather from HostBuffer
    # ------------------------------------------------------------------

    def async_gather(
        self,
        phase,
        pid: int,
        host_buffer: HostBuffer,
        boundaries: List[Optional[Tensor]],
        stream: torch.cuda.Stream,
    ) -> None:
        """Allocate GPU tensor and gather from host partitions.

        Combines allocate() + host_buffer.async_gather() for convenience.
        """
        self.allocate(pid)
        host_buffer.async_gather(phase, pid, self._tensors[pid], boundaries, stream)

    def async_gather_direct(
        self,
        phase,
        pid: int,
        gpu_target: Tensor,
        boundaries: List[Optional[Tensor]],
        stream: torch.cuda.Stream, 
    ) -> None:
        self.allocate(pid)


        import os

        file_prefix = f"feat_l0_"

        file_paths = [
            f"/mnt/nvme/feat_l0_p{i}.pt"
            for i in range(self.num_parts)
        ]

        num_nodes = []
        for path in file_paths:
            f_bytes = os.path.getsize(path)
            total_features = f_bytes // (gpu_target.size(1) * gpu_target.element_size())
            num_nodes.append(total_features)

        assert gpu_target.is_cuda, "gpu_target must be a CUDA tensor"

        bndries = []
        for i in range(self.num_parts):
            if i == pid or boundaries[i] is None:
                bndries.append(torch.empty(0, dtype=torch.long, device="cpu"))
            else:
                bndries.append(boundaries[i].cpu())

        stream.wait_stream(torch.cuda.current_stream(gpu_target.device))

        # Tensor metrics
        feature_dim = gpu_target.size(1)
        bytes_per_elem = gpu_target.element_size()
        row_bytes = feature_dim * bytes_per_elem
        bytes_to_gb = 1024**3

        # Record gather size statistics
        target_partition_nodes = num_nodes[pid]
        boundary_nodes = sum(b.numel() for b in bndries)
        target_partition_gb = (target_partition_nodes * row_bytes) / bytes_to_gb
        boundary_gb = (boundary_nodes * row_bytes) / bytes_to_gb
        stat.add_actual_size(target_partition_gb, target_partition_gb + boundary_gb)

        import time

        t0 = time.perf_counter_ns()
        with torch.cuda.stream(stream):
            if self._ops is not None: 
                print("Async loading")
                self._ops.gather_partitions_gds(pid, file_paths, num_nodes, gpu_target, bndries)
            else:
                # Load full target partition --> whole file
                print("Fallback")
                exit(1)
                offset = num_nodes[pid]
                self._backend.gpu_read(
                    status=3, # read whole file and close
                    fd=None,
                    file_id=f"{self._backend._storage_dir}/{self._file_prefix}_p{pid}.pt",
                    tensor=gpu_target[:offset],
                    file_offset=0,
                    stream=stream,
                )

                print("Target partition loaded")

                part_streams = []

                for i in range(self.num_parts):
                    if i == pid or bndries[i].numel() == 0:
                        continue

                    part_file = f"{self._backend._storage_dir}/{self._file_prefix}_p{i}.pt"

                    p_stream = torch.cuda.Stream(device=gpu_target.device)
                    p_stream.wait_stream(stream)
                    part_streams.append(p_stream)

                    index = 0
                    chunks = []
                    while index < len(bndries[i]):
                        start = index
                        end = index + 1

                        while end < len(bndries[i]) and bndries[i][end] == bndries[i][end - 1] + 1:
                            end += 1
                        
                        chunk_len = end - start
                        file_offset = bndries[i][start].item() * row_bytes
                        chunks.append((file_offset, chunk_len))

                        index = end

                    fd = None
                    total_chunks = len(chunks)
                    curr_offset = offset

                    with torch.cuda.stream(p_stream):
                        for chunk_idx, (file_offset, chunk_len) in enumerate(chunks):
                            if chunk_idx == 0 and total_chunks == 1:
                                # single read --> open, read, close
                                fd = self._backend.gpu_read(
                                    status=3,
                                    fd=None,
                                    file_id=part_file,
                                    tensor=gpu_target[curr_offset : curr_offset + chunk_len],
                                    file_offset=file_offset,
                                    stream=p_stream,
                                )

                            elif chunk_idx == 0:
                                # first read --> open file
                                fd = self._backend.gpu_read(
                                    status=0,
                                    fd=None,
                                    file_id=part_file,
                                    tensor=gpu_target[curr_offset : curr_offset + chunk_len],
                                    file_offset=file_offset,
                                    stream=p_stream,
                                )

                            elif chunk_idx != total_chunks - 1:
                                # file already open
                                fd = self._backend.gpu_read(
                                    status=1,
                                    fd=fd,
                                    file_id=part_file,
                                    tensor=gpu_target[curr_offset : curr_offset + chunk_len],
                                    file_offset=file_offset,
                                    stream=p_stream,
                                )

                            else:
                                # last read --> close file and record timestamps
                                fd = self._backend.gpu_read(
                                    status=2,
                                    fd=fd,
                                    file_id=part_file,
                                    tensor=gpu_target[curr_offset : curr_offset + chunk_len],
                                    file_offset=file_offset,
                                    stream=p_stream,
                                )

                            curr_offset += chunk_len
                        
                    offset += sum(c[1] for c in chunks)
                    print(f"Boundary partition {i} data loaded")

                for p_stream in part_streams:
                    stream.wait_stream(p_stream)

        tn = time.perf_counter_ns()
        stat.load_GPU_timestamp(phase, "copy", t0, tn)

        bt = gpu_target.numel() * bytes_per_elem
        gb = round(bt / bytes_to_gb, 3)

        #host_buffer.async_gather_direct(phase, pid, self._tensors[pid], boundaries, stream)

    def h2d_synchronize(self, stream: torch.cuda.Stream) -> None:
        """Wait for H2D operations to complete.

        Two-phase sync: C++ thread pool + CUDA stream.
        """
        try:
            from grinnder._C import h2d_synchronize as _h2d_sync
            try:
                _h2d_sync()
            except RuntimeError:
                pass  # No pending H2D operations
        except ImportError:
            pass
        torch.cuda.synchronize(stream)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Free all GPU memory."""
        for i in range(self.num_parts):
            self.release(i)
