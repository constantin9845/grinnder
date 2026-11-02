"""Host buffer: partition-wise tensor management on host RAM / NVMe storage.

Each HostBuffer manages one "layer" of activations or gradients, split into
per-partition tensors. Supports async gather, scatter, fill, upload, and
bypass-to-storage operations via C++ extensions.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import time
import os
from torch import Tensor

from grinnder.storage.backend import StorageBackend
from grinnder.stats import stat


def _load_ops():
    """Lazy-load C++ extension ops."""
    try:
        import grinnder._C as _C
        return _C
    except ImportError:
        return None


def _empty_host_tensor(
    rows: int,
    cols: int,
    *,
    pin_memory: bool,
    allocate: bool,
) -> Tensor:
    """Allocate a CPU tensor with optional pinned-memory allocation."""
    shape = (rows, cols) if allocate else (0, cols)
    try:
        return torch.empty(
            shape,
            dtype=torch.float32,
            pin_memory=pin_memory,
        )
    except RuntimeError:
        if not pin_memory:
            raise
        return torch.empty(shape, dtype=torch.float32)


class PendingRead:
    """Async SSD -> GPU read: one or more gather jobs, plus an optional step
    (e.g. summing gradient contributions) that runs once they finished."""

    def __init__(self, ops, jobs: List[Optional[int]], post=None):
        self._ops = ops
        self._jobs = [j for j in jobs if j is not None]
        self._post = post

    def wait(self) -> None:
        while self._jobs:
            self._ops.gather_wait(self._jobs.pop(0))
        post, self._post = self._post, None
        if post is not None:
            post()


def _empty_pinned_host_cache() -> None:
    """Release cached pinned-host blocks when PyTorch exposes the hook."""
    empty_cache = getattr(torch._C, "_host_emptyCache", None)
    if empty_cache is not None:
        empty_cache()


class HostBuffer:
    """Partition-wise tensor storage on host memory.

    Stores one tensor per partition. Supports async transfers to/from GPU
    and to/from NVMe storage. HostBuffer uses pageable CPU tensors by default;
    pass ``pin_memory=True`` to request pinned CPU tensors.

    Layout for gather:
        GPU tensor = [intra(pid) | boundary_from_p0 | ... | boundary_from_pN]
        - intra: contiguous memcpy from self[pid]
        - boundary: index_select from self[src_pid] at boundary indices

    Args:
        num_parts: Number of partitions.
        part_sizes: Number of nodes in each partition.
        embedding_dim: Feature dimension.
        backend: StorageBackend for NVMe I/O (None = host-only).
        file_prefix: Prefix for NVMe file IDs.
    """

    def __init__(
        self,
        num_parts: int,
        part_sizes: List[int],
        embedding_dim: int,
        backend: Optional[StorageBackend] = None,
        file_prefix: str = "host",
        lazy: bool = True,
        pin_memory: bool = False,
    ):
        self.num_parts = num_parts
        self.part_sizes = part_sizes
        self.embedding_dim = embedding_dim
        self._backend = backend
        self._file_prefix = file_prefix
        self._element_size = torch.tensor([], dtype=torch.float32).element_size()
        self._lazy = lazy
        self._pin_memory = pin_memory
        self._zero_initialized = [False] * num_parts

        # GPU <-> Storage: kvikio required
        try:
            import kvikio
            self._kvikio = kvikio
        except ImportError:
            raise ImportError(
                "kvikio is required for GriNNder's grinnder mode. "
                "Install with: pip install kvikio-cu12"
            )

        # Tensor objects keep their logical shape, while storage is allocated
        # only for partitions resident in host memory.
        self._tensors: List[Tensor] = []
        for i in range(num_parts):
            t = _empty_host_tensor(
                part_sizes[i],
                embedding_dim,
                pin_memory=pin_memory,
                allocate=not lazy,
            )
            if not lazy:
                t.zero_()
            self._tensors.append(t)

        self._ops = _load_ops()

        # pid -> PendingRead of an SSD -> GPU read started by async_read_raw
        self._pending_reads: Dict[int, "PendingRead"] = {}

        # Gradient buffers (see begin_accumulation). Each source partition's
        # whole input gradient ([intra | boundary_from_p0 | ...]) is written
        # once as its own SSD tensor "<prefix>_from<src>". _contrib[src] says
        # where each destination's rows sit in it:
        #   dst -> (first row in the blob, number of rows, dst row indices or None)
        self._accumulate = False
        self._contrib: Dict[int, Tuple[str, Dict[int, Tuple[int, int, Optional[Tensor]]]]] = {}
        # file ids of contribution writes still in flight
        self._pending_contrib: List[str] = []


    def __getitem__(self, pid: int) -> Tensor:
        """Get host tensor for partition pid, allocating it if needed."""
        self.allocate(pid)
        return self._tensors[pid]

    def __len__(self) -> int:
        return self.num_parts

    def partition_nbytes(self, pid: int) -> int:
        """Bytes required by one partition when resident in host RAM."""
        return self.part_sizes[pid] * self.embedding_dim * self._element_size

    @property
    def total_nbytes(self) -> int:
        return sum(self.partition_nbytes(pid) for pid in range(self.num_parts))

    def allocate(self, pid: int) -> Tensor:
        """Allocate host storage for partition pid if it is not resident."""
        t = self._tensors[pid]
        expected = self.partition_nbytes(pid)
        if t.untyped_storage().size() < expected:
            t.untyped_storage().resize_(expected)
        expected_shape = (self.part_sizes[pid], self.embedding_dim)
        if tuple(t.shape) != expected_shape:
            t.set_(t.untyped_storage(), 0, expected_shape, (self.embedding_dim, 1))
        return t

    def release(self, pid: int) -> None:
        """Release host storage for one partition while preserving tensor metadata."""
        self._tensors[pid].untyped_storage().resize_(0)
        self._zero_initialized[pid] = False
        if self._pin_memory:
            _empty_pinned_host_cache()

    def release_all(self) -> None:
        """Release host storage for every partition."""
        for pid in range(self.num_parts):
            self.release(pid)

    def is_allocated(self, pid: int) -> bool:
        """Return whether partition pid currently occupies host storage."""
        return self._tensors[pid].untyped_storage().size() >= self.partition_nbytes(pid)

    def allocated_pids(self) -> List[int]:
        """Return partitions that currently occupy host storage."""
        return [pid for pid in range(self.num_parts) if self.is_allocated(pid)]

    def zero_partition(self, pid: int) -> None:
        """Allocate and zero one partition buffer."""
        self.allocate(pid)
        self._tensors[pid].zero_()
        self._zero_initialized[pid] = True

    @property
    def resident_bytes(self) -> int:
        """Current host bytes occupied by resident partition storage."""
        return sum(
            self.partition_nbytes(pid)
            for pid in range(self.num_parts)
            if self.is_allocated(pid)
        )

    def storage_exists(self, pid: int) -> bool:
        """Return whether a backing storage copy exists for one partition."""
        if self._backend is None:
            return False
        # Data lives on the raw device; the FTL mapping is the source of truth.
        return self._file_id(pid) in self._backend.host_ftl.tensor_table

    def ensure_storage_copy(self, pid: int) -> None:
        """Write a resident partition to storage if no backing copy exists."""
        if self._backend is None or self.storage_exists(pid):
            return
        if not self.is_allocated(pid):
            raise RuntimeError(
                f"HostBuffer partition {pid} is not resident for storage write"
            )
        self.cpu_to_storage(pid)

    # ------------------------------------------------------------------
    # GPU -> Host (D2H)
    # ------------------------------------------------------------------

    def async_fill(
        self, pid: int, gpu_data: Tensor, stream: torch.cuda.Stream
    ) -> None:
        """D2H: copy GPU tensor to host partition buffer.

        Args:
            pid: Partition index.
            gpu_data: GPU tensor [part_sizes[pid], dim].
            stream: CUDA stream for async copy.
        """
        assert gpu_data.is_cuda
        self.allocate(pid)
        stream.wait_stream(torch.cuda.current_stream(gpu_data.device))
        with torch.cuda.stream(stream):
            if self._ops is not None:
                self._ops.d2h_copy_async(gpu_data, self._tensors[pid])
            else:
                self._tensors[pid].copy_(gpu_data, non_blocking=True)
        self._zero_initialized[pid] = True

    def d2h_synchronize(self, stream: torch.cuda.Stream) -> None:
        """Wait for D2H operations to complete.

        Two-phase synchronization:
        1. C++ thread pool sync: wait for cudaMemcpyAsync calls to be submitted
        2. CUDA stream sync: wait for all submitted CUDA ops to complete
        Also waits for gradient row writes started by accumulate_raw.
        """
        self.wait_accumulations()
        if self._ops is not None:
            try:
                self._ops.d2h_synchronize()
            except RuntimeError:
                pass  # No pending D2H operations
        torch.cuda.synchronize(stream)

    # ------------------------------------------------------------------
    # Host -> GPU (H2D)
    # ------------------------------------------------------------------

    def async_upload(
        self, phase, pid: int, gpu_target: Tensor, stream: torch.cuda.Stream
    ) -> None:
        """H2D: copy host partition buffer to GPU tensor.

        Args:
            pid: Partition index.
            gpu_target: Pre-allocated GPU tensor.
            stream: CUDA stream for async copy.
        """
        assert gpu_target.is_cuda
        if not self.is_allocated(pid):
            raise RuntimeError(
                f"HostBuffer partition {pid} is not resident for upload"
            )
        stream.wait_stream(torch.cuda.current_stream(gpu_target.device))
        
        t0 = time.perf_counter_ns()
        with torch.cuda.stream(stream):
            if self._ops is not None:
                self._ops.h2d_copy_async(self._tensors[pid], gpu_target)
            else:
                gpu_target.copy_(self._tensors[pid], non_blocking=True)

        tn = time.perf_counter_ns()

        stat.load_GPU_timestamp("gradient", "copy", t0, tn) 



    def h2d_synchronize(
        self, stream: torch.cuda.Stream, pid: Optional[int] = None
    ) -> None:
        """Wait for H2D operations to complete.

        Two-phase synchronization:
        1. C++ thread pool sync: wait for cudaMemcpyAsync calls to be submitted
        2. CUDA stream sync: wait for all submitted CUDA ops to complete
        Also waits for async_read_raw reads (of pid only, if given).
        """
        self.wait_reads(pid)
        if self._ops is not None:
            try:
                self._ops.h2d_synchronize()
            except RuntimeError:
                pass  # No pending H2D operations
        torch.cuda.synchronize(stream)

    # ------------------------------------------------------------------
    # Gather: multiple host partitions -> one GPU tensor
    # ------------------------------------------------------------------

    def async_gather(
        self,
        phase,
        pid: int,
        gpu_target: Tensor,
        boundaries: List[Optional[Tensor]],
        stream: torch.cuda.Stream,
    ) -> None:
        """Gather features from host partitions to GPU.

        GPU layout: [intra(pid) | boundary_from_p0 | ... | boundary_from_pN]

        Args:
            pid: Target partition index.
            gpu_target: Pre-allocated GPU tensor [total_size, dim].
            boundaries: boundaries[src_pid] = index tensor into self[src_pid].
                        boundaries[pid] = None (intra-partition, copied contiguously).
            stream: CUDA stream.
        """
        t0 = time.perf_counter_ns()
        assert gpu_target.is_cuda
        required_pids = {pid}
        for src_pid, boundary in enumerate(boundaries):
            if src_pid != pid and boundary is not None and boundary.numel() > 0:
                required_pids.add(src_pid)
        missing = [src_pid for src_pid in required_pids if not self.is_allocated(src_pid)]
        if missing:
            raise RuntimeError(
                f"HostBuffer partitions {missing} are not resident for gather"
            )

        # Build boundary list (replace None with empty tensor)
        bound_size = 0
    
        bndries = []
        for i in range(self.num_parts):
            if i == pid or boundaries[i] is None:
                bndries.append(torch.empty(0, dtype=torch.long))
            else:
                bndries.append(boundaries[i])


        stream.wait_stream(torch.cuda.current_stream(gpu_target.device))

        tn = time.perf_counter_ns()
        stat.load_GPU_timestamp(phase, "gather", t0, tn)

        target_partition_nodes = self._tensors[pid].size(0)
        boundary_nodes = sum(b.numel() for b in bndries)

        feature_dim = gpu_target.size(1)
        bytes_per_elem = gpu_target.element_size()
        row_bytes = feature_dim * bytes_per_elem
        bytes_to_gb = 1024**3

        target_partition_gb = (target_partition_nodes * feature_dim * bytes_per_elem) / bytes_to_gb
        boundary_gb = (boundary_nodes * feature_dim * bytes_per_elem) / bytes_to_gb

        stat.add_actual_size(target_partition_gb, target_partition_gb + boundary_gb)

        boundary_nodes = sum(b.numel() for b in bndries)
        total_boundary_parts_nodes = sum(
            self._tensors[src_pid].size(0) 
            for src_pid, bnd in enumerate(bndries) 
            if src_pid != pid and bnd.numel() > 0
        )

        overall_pct = (boundary_nodes / total_boundary_parts_nodes * 100) if total_boundary_parts_nodes > 0 else 0.0
        stat.add_boundary_utilization(overall_pct)

        t0 = time.perf_counter_ns()
        with torch.cuda.stream(stream):
            if self._ops is not None and 2 == 3:
                self._ops.gather_partitions(pid, self._tensors, gpu_target, bndries)
                print("Not a fallback")
            else:
                # Python fallback
                print("Fallback")
                offset = self._tensors[pid].size(0)

                with torch.no_grad():
                    gpu_target[:offset].copy_(self._tensors[pid], non_blocking=True)

                    for i in range(self.num_parts):
                        if i == pid or bndries[i].numel() == 0:
                            continue
                        selected = self._tensors[i].index_select(0, bndries[i])
                        n = selected.size(0)
                        gpu_target[offset : offset + n].copy_(selected, non_blocking=True)
                        offset += n
        
        tn = time.perf_counter_ns()
        stat.load_GPU_timestamp(phase, "copy", t0, tn)

        bt = gpu_target.numel() * gpu_target.element_size()
        gb = round(bt / (1024**3),3)

        print(f"\tPartition {pid} loads {gb} GB from other partitions")



    def open_files(self, num_parts):

        file_paths = [
            f"{self._backend._storage_dir}/{self._file_prefix}_p{i}.pt"
            for i in range(num_parts)
        ]
        
        fds = self._ops.open_files(file_paths)
        return fds
    
    def close_files(self, fds):
        self._ops.close_files(fds)

    # ------------------------------------------------------------------
    # Gather Direct: target partition + boundary feat/act from NVMe -> one GPU tensor [File based]
    # ------------------------------------------------------------------

    def async_gather_direct(
    self,
    phase,
    fds,
    pid: int,
    gpu_target: Tensor,
    boundaries: List[Optional[Tensor]],
    stream: torch.cuda.Stream,
) -> None:
        """Gather features directly from NVMe files to GPU VRAM via GDS.

        GPU layout: [intra(pid) | boundary_from_p0 | ... | boundary_from_pN]

        Args:
            phase: Current execution phase name for stats.
            pid: Target partition index.
            gpu_target: Pre-allocated CUDA tensor [total_nodes, feature_dim].
            boundaries: boundaries[src_pid] = index tensor into partition src_pid.
                        boundaries[pid] = None (intra-partition, copied contiguously).
            stream: CUDA stream for non-blocking asynchronous execution.
        """
        import os

        t0 = time.perf_counter_ns()

        assert gpu_target.is_cuda, "gpu_target must be a CUDA tensor"
        assert gpu_target.is_contiguous(), "gpu_target must be contiguous"

        # Build boundary list (replace None with empty tensor)
        bndries = []
        for i in range(self.num_parts):
            if i == pid or boundaries[i] is None:
                bndries.append(torch.empty(0, dtype=torch.long))
            else:
                bndries.append(boundaries[i])

        stream.wait_stream(torch.cuda.current_stream(gpu_target.device))

        # File paths
        file_paths = [
            f"{self._backend._storage_dir}/{self._file_prefix}_p{i}.pt"
            for i in range(self.num_parts)
        ]

        # Tensor metrics
        feature_dim = gpu_target.size(1)
        bytes_per_elem = gpu_target.element_size()
        row_bytes = feature_dim * bytes_per_elem
        target_partition_nodes = self._tensors[pid].size(0)

        bytes_to_gb = 1024**3

        # Determine number of rows in each file.
        # This does not load the files.
        file_sizes = [
            os.path.getsize(path) if os.path.exists(path) else 0
            for path in file_paths
        ]

        num_nodes = [
            file_sizes[i] // row_bytes
            for i in range(self.num_parts)
        ]

        # Check that files have valid row-aligned sizes
        for i in range(self.num_parts):
            if file_sizes[i] == 0:
                raise RuntimeError(
                    f"Partition file missing or empty: "
                    f"pid={i}, file={file_paths[i]}"
                )

            if file_sizes[i] % row_bytes != 0:
                raise RuntimeError(
                    f"Partition file size is not divisible by row_bytes:\n"
                    f"  pid={i}\n"
                    f"  file={file_paths[i]}\n"
                    f"  file_size={file_sizes[i]}\n"
                    f"  row_bytes={row_bytes}"
                )

        # Record gather size statistics
        target_partition_nodes = num_nodes[pid]
        boundary_nodes = sum(b.numel() for b in bndries)

        target_partition_gb = (
            target_partition_nodes * row_bytes
        ) / bytes_to_gb

        boundary_gb = (
            boundary_nodes * row_bytes
        ) / bytes_to_gb

        stat.add_actual_size(
            target_partition_gb,
            target_partition_gb + boundary_gb,
        )

        total_boundary_parts_nodes = sum(
            num_nodes[src_pid]
            for src_pid, bnd in enumerate(bndries)
            if src_pid != pid and bnd.numel() > 0
        )

        overall_pct = (
            boundary_nodes / total_boundary_parts_nodes * 100
            if total_boundary_parts_nodes > 0
            else 0.0
        )

        stat.add_boundary_utilization(overall_pct)

        # Check GPU target size
        required_nodes = target_partition_nodes + boundary_nodes

        if gpu_target.size(0) < required_nodes:
            raise RuntimeError(
                f"gpu_target is too small:\n"
                f"  target pid={pid}\n"
                f"  target_partition_nodes={target_partition_nodes}\n"
                f"  boundary_nodes={boundary_nodes}\n"
                f"  required_nodes={required_nodes}\n"
                f"  gpu_target.size(0)={gpu_target.size(0)}"
            )

        with torch.cuda.stream(stream):

            if self._ops is not None:
                self._ops.gather_partitions_direct(pid, fds, gpu_target, bndries)
                print("Not a fallback")

            else:

                # --------------------------------------------------------
                # Fallback
                # --------------------------------------------------------

                print("Fallback")

                offset = num_nodes[pid]

                # read full target partition fill
                self._backend.gpu_read_direct(
                    status=0,
                    fd=None,
                    file_id=f"{self._backend._storage_dir}/{self._file_prefix}_p{pid}",
                    tensor=gpu_target[:offset],
                    offset=0,
                    stream=stream,
                )

                print("Target partition loaded")

                # --------------------------------------------------------
                # Load only required boundary file rows
                # --------------------------------------------------------

                for i in range(self.num_parts):

                    if i == pid or bndries[i].numel() == 0:
                        continue

                    part_file = file_paths[i]
                    indices = bndries[i]
                    num_rows = indices.size(0)
                    part_file_size = file_sizes[i]
                    max_valid_nodes = num_nodes[i]

                    if indices.device.type != "cpu":
                        indices = indices.cpu()


                    fd = None

                    # iterate nodes from partition
                    for k in range(num_rows):

                        node_idx = indices[k].item()
                        file_offset = node_idx * row_bytes

                        # Exactly one output row, preserving index_select order.
                        dest_row = gpu_target[
                            offset : offset + 1
                        ]

                        if k == 0 and num_rows == 1:
                            # Single row: open, read, close
                            fd = self._backend.gpu_read_direct(
                                status=0,
                                fd=None,
                                file_id=f"{self._backend._storage_dir}/{self._file_prefix}_p{i}",
                                tensor=dest_row,
                                offset=file_offset,
                                stream=stream,
                            )

                        elif k == 0:
                            # First row: open handle + read
                            fd = self._backend.gpu_read_direct(
                                status=1,
                                fd=None,
                                file_id=f"{self._backend._storage_dir}/{self._file_prefix}_p{i}",
                                tensor=dest_row,
                                offset=file_offset,
                                stream=stream,
                            )

                        elif k != num_rows - 1:
                            # Middle row: use persistent fd
                            fd = self._backend.gpu_read_direct(
                                status=2,
                                fd=fd,
                                file_id=f"{self._backend._storage_dir}/{self._file_prefix}_p{i}",
                                tensor=dest_row,
                                offset=file_offset,
                                stream=stream,
                            )

                        else:
                            # Last row: read + close
                            fd = self._backend.gpu_read_direct(
                                status=3,
                                fd=fd,
                                file_id=f"{self._backend._storage_dir}/{self._file_prefix}_p{i}",
                                tensor=dest_row,
                                offset=file_offset,
                                stream=stream,
                            )

                        offset += 1

                    print(f"Boundary partition {i} data loaded")

                # --------------------------------------------------------
                # Final sanity check
                # --------------------------------------------------------

                expected_offset = (
                    target_partition_nodes +
                    boundary_nodes
                )

                if offset != expected_offset:
                    raise RuntimeError(
                        f"Gather offset mismatch:\n"
                        f"  final offset = {offset}\n"
                        f"  expected = {expected_offset}"
                    )


        bt = gpu_target.numel() * gpu_target.element_size()
        gb = round(bt / (1024**3), 3)

        print(
            f"\tPartition {pid} loads "
            f"{gb} GB from other partitions"
        )

    # ------------------------------------------------------------------
    # Gather Direct: target partition + boundary feat/act from NVMe -> one GPU tensor [Raw block device + host FTL]
    # ------------------------------------------------------------------

    def async_gather_direct_raw(
        self,
        phase,
        fds,
        pid: int,
        gpu_target: Tensor,
        boundaries: List[Optional[Tensor]],
        stream: torch.cuda.Stream,
    ) -> None:
        """Gather features directly from NVMe files to GPU VRAM via GDS.

        GPU layout: [intra(pid) | boundary_from_p0 | ... | boundary_from_pN]
        """
        assert gpu_target.is_cuda, "gpu_target must be a CUDA tensor"
        assert gpu_target.is_contiguous(), "gpu_target must be contiguous"

        row_bytes = gpu_target.size(1) * gpu_target.element_size()

        # Activations of this layer may still be in flight to the SSD.
        self.wait_storage_writes()

        segments = self._gather_segments(pid, boundaries, gpu_target.size(0))
        dev_offs = self._segment_dev_offsets(segments, row_bytes)
        total_rows = len(dev_offs)
        assert total_rows <= gpu_target.size(0), (
            f"Gather needs {total_rows} rows but gpu_target has {gpu_target.size(0)}"
        )
        runs_t = self._merge_runs(dev_offs, np.arange(total_rows, dtype=np.int64), row_bytes)

        print(
            f"[gather_raw] pid={pid} target_rows={len(segments[0][1])} "
            f"boundary_rows={total_rows - len(segments[0][1])} runs={runs_t.size(0)}"
        )

        # The reads start in the background once `stream` reaches this point,
        # i.e. after everything already queued on the current stream (so memory
        # reused from a freed tensor is no longer in use).
        stream.wait_stream(torch.cuda.current_stream(gpu_target.device))
        with torch.cuda.stream(stream):
            job = self._submit_read(gpu_target, runs_t)

        bt = gpu_target.numel() * gpu_target.element_size()
        gb = round(bt / (1024**3), 3)

        print(
            f"\tPartition {pid} loads "
            f"{gb} GB from other partitions"
        )
        # Job id: gpu_target holds the data only after self._ops.gather_wait(job).
        return job

    # ------------------------------------------------------------------
    # SSD <-> GPU helpers (host FTL addressing, host memory only as staging)
    # ------------------------------------------------------------------

    def _file_id(self, pid: int) -> str:
        return f"{self._file_prefix}_p{pid}"

    def _gather_segments(
        self, pid: int, boundaries: List[Optional[Tensor]], gpu_rows: int
    ) -> List[Tuple[int, np.ndarray]]:
        """Rows of each partition, in GPU layout order:
        [intra(pid) | boundary_from_p0 | ... | boundary_from_pN]."""
        num_boundary_rows = sum(
            b.numel() for j, b in enumerate(boundaries) if j != pid and b is not None
        )
        num_target_rows = (
            boundaries[pid].numel()
            if boundaries[pid] is not None
            else gpu_rows - num_boundary_rows
        )
        segments = [(pid, np.arange(num_target_rows, dtype=np.int64))]
        for i in range(self.num_parts):
            if i != pid and boundaries[i] is not None and boundaries[i].numel() > 0:
                segments.append((i, boundaries[i].cpu().to(torch.long).numpy()))
        return segments

    def _segment_dev_offsets(
        self, segments: List[Tuple[int, np.ndarray]], row_bytes: int
    ) -> np.ndarray:
        """Row -> device byte offset, using the FTL's current mapping
        (dev_lba * sector_size + offset in page)."""
        ftl = self._backend.host_ftl
        dev_offs = [
            ftl.get_row_file_offsets(self._file_id(i), rows, row_bytes)
            for i, rows in segments
            if len(rows) > 0
        ]
        if not dev_offs:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(dev_offs).astype(np.int64)

    @staticmethod
    def _merge_runs(dev_offs: np.ndarray, gpu_rows: np.ndarray, row_bytes: int) -> Tensor:
        """Merge rows that are back to back both on the device and in the GPU
        tensor into runs of (dev_off, gpu_off, size), in bytes."""
        n = len(dev_offs)
        if n == 0:
            return torch.empty((0, 3), dtype=torch.int64)
        breaks = np.flatnonzero(
            (np.diff(dev_offs) != row_bytes) | (np.diff(gpu_rows) != 1)
        ) + 1
        starts = np.concatenate(([0], breaks))
        lens = np.diff(np.concatenate((starts, [n])))
        runs = np.stack(
            [dev_offs[starts], gpu_rows[starts] * row_bytes, lens * row_bytes], axis=1
        ).astype(np.int64)
        return torch.from_numpy(runs)

    def _submit_read(self, gpu_target: Tensor, runs_t: Tensor) -> Optional[int]:
        """Start SSD -> GPU reads of runs into gpu_target (current stream orders
        the start). Returns a job id for gather_wait, or None if nothing to read."""
        if runs_t.size(0) == 0:
            return None
        ftl = self._backend.host_ftl
        if os.environ.get("GRINNDER_GDS_DIRECT") == "1":
            # cuFile batch reads (real GDS path)
            return self._ops.gather_partitions_direct_raw_async(
                ftl.cuFD, gpu_target, runs_t, ftl.sector_size
            )
        # NVMe reads into pinned host memory + copies to the GPU, no cuFile
        return self._ops.gather_partitions_nvme_async(
            ftl.device_fd, gpu_target, runs_t, ftl.sector_size
        )

    def _submit_write(self, gpu_src: Tensor, runs_t: Tensor) -> Optional[int]:
        """Start GPU -> SSD writes of runs from gpu_src (current stream orders
        the start). Returns a job id for gpu_write_wait, or None."""
        if runs_t.size(0) == 0:
            return None
        ftl = self._backend.host_ftl
        if os.environ.get("GRINNDER_GDS_DIRECT") == "1":
            return self._ops.gpu_write_async(
                ftl.cuFD, gpu_src, runs_t, ftl.sector_size, ftl.page_size
            )
        return self._ops.gpu_write_nvme_async(
            ftl.device_fd, gpu_src, runs_t, ftl.sector_size, ftl.page_size
        )

    def _blob_row_runs(
        self, file_id: str, first_row: int, n: int, dst_row: int, row_bytes: int
    ) -> Tensor:
        """Runs reading rows [first_row, first_row + n) of SSD tensor file_id
        into rows [dst_row, dst_row + n) of a GPU tensor."""
        rows = np.arange(first_row, first_row + n, dtype=np.int64)
        dev_offs = self._backend.host_ftl.get_row_file_offsets(file_id, rows, row_bytes)
        return self._merge_runs(
            dev_offs.astype(np.int64),
            np.arange(dst_row, dst_row + n, dtype=np.int64),
            row_bytes,
        )

    # ------------------------------------------------------------------
    # Whole partition: SSD -> GPU
    # ------------------------------------------------------------------

    def async_read_raw(
        self, pid: int, gpu_target: Tensor, stream: torch.cuda.Stream, track: bool = True
    ) -> "PendingRead":
        """Read partition pid from the SSD into gpu_target [part_sizes[pid], dim].

        Same path as the forward gather (FTL addresses, NVMe reads, host
        staging only).
        - Activation buffers: the whole tensor, using its FTL page runs.
        - Gradient buffers (after begin_accumulation): pid's own contribution
          plus the rows other partitions wrote for pid, summed on the GPU.

        track=True: wait with h2d_synchronize / wait_reads(pid).
        track=False: the caller waits on the returned PendingRead.
        """
        assert gpu_target.is_cuda and gpu_target.is_contiguous()
        assert gpu_target.size(0) == self.part_sizes[pid]

        self.wait_reads(pid)
        self.wait_storage_writes(pid)
        self.wait_accumulations()

        t0 = time.perf_counter_ns()
        stream.wait_stream(torch.cuda.current_stream(gpu_target.device))
        if self._accumulate:
            pending = self._read_gradient(pid, gpu_target, stream)
        else:
            # Whole tensor: the FTL's page runs, as written. Works for any row
            # size (e.g. 76-byte logits rows); the last page's padding is
            # read but not copied.
            runs = self._backend.host_ftl.get_tensor_io_runs(self._file_id(pid))
            runs_t = torch.tensor(runs, dtype=torch.int64).reshape(-1, 3)
            with torch.cuda.stream(stream):
                job = self._submit_read(gpu_target, runs_t)
            pending = PendingRead(self._ops, [job])
            print(f"[read_raw] {self._file_id(pid)} runs={runs_t.size(0)}")
        tn = time.perf_counter_ns()
        stat.load_GPU_timestamp("gradient", "copy", t0, tn)

        if track:
            self._pending_reads[pid] = pending
        return pending

    def wait_reads(self, pid: Optional[int] = None) -> None:
        """Wait for async_read_raw reads of one partition (or all)."""
        if not self._pending_reads:
            return
        pids = list(self._pending_reads) if pid is None else [pid]
        for p in pids:
            pending = self._pending_reads.pop(p, None)
            if pending is not None:
                pending.wait()

    # ------------------------------------------------------------------
    # Gradients on the SSD: write each contribution once, sum on read
    # ------------------------------------------------------------------

    def begin_accumulation(self) -> None:
        """Start a backward pass for this gradient buffer. Contributions of
        the previous pass are forgotten (their FTL mappings are reused)."""
        self.wait_accumulations()
        self._accumulate = True
        self._contrib = {}

    def accumulate_raw(
        self,
        pid: int,
        gpu_grad: Tensor,
        boundaries: List[Optional[Tensor]],
        stream: torch.cuda.Stream,
    ) -> None:
        """Store partition pid's input gradient (gathered layout
        [intra(pid) | boundary_from_p0 | ...]) on the SSD as one tensor.

        No read before the write: the sum per destination partition happens
        when it is read (_read_gradient). The write runs in the background;
        wait_accumulations() / d2h_synchronize() waits for it.
        """
        assert self._backend is not None and self._accumulate, (
            "call begin_accumulation() before accumulate_raw()"
        )
        assert gpu_grad.is_cuda and gpu_grad.is_contiguous()

        # Keep at most one contribution write in flight (bounds GPU memory
        # held by pending writes); it had the previous partition's whole
        # backward to finish.
        self.wait_accumulations()

        # Where each destination partition's rows sit in this gradient.
        segs: Dict[int, Tuple[int, int, Optional[Tensor]]] = {}
        num_boundary = sum(
            b.numel() for j, b in enumerate(boundaries) if j != pid and b is not None
        )
        n_intra = gpu_grad.size(0) - num_boundary
        segs[pid] = (0, n_intra, None)
        off = n_intra
        for j in range(self.num_parts):
            b = boundaries[j]
            if j != pid and b is not None and b.numel() > 0:
                segs[j] = (off, b.numel(), b)
                off += b.numel()
        assert off == gpu_grad.size(0), (
            f"Gradient has {gpu_grad.size(0)} rows, layout needs {off}"
        )

        file_id = f"{self._file_prefix}_from{pid}"
        stream.wait_stream(torch.cuda.current_stream(gpu_grad.device))
        # Same write path as forward activations (FTL, mapping reused when
        # the size matches). The job keeps gpu_grad alive until it is done.
        self._backend.gpu_write(gpu_grad, file_id, stream, async_=True)
        self._pending_contrib.append(file_id)
        self._contrib[pid] = (file_id, segs)

    def wait_accumulations(self) -> None:
        """Wait for in-flight contribution writes of this buffer."""
        if self._pending_contrib:
            self._backend.wait_gpu_writes(file_ids=self._pending_contrib)
            self._pending_contrib = []

    def _read_gradient(
        self, pid: int, gpu_target: Tensor, stream: torch.cuda.Stream
    ) -> "PendingRead":
        """Gradient of partition pid = its own contribution (all its rows)
        + the boundary rows every other partition wrote for it.

        Own rows are read straight into gpu_target; the other rows are read
        (contiguous per source) into a staging tensor and index_add-ed into
        gpu_target once the reads finished (PendingRead.wait)."""
        row_bytes = gpu_target.size(1) * gpu_target.element_size()
        own = self._contrib.get(pid)
        others = [
            (file_id, segs[pid])
            for src, (file_id, segs) in self._contrib.items()
            if src != pid and pid in segs
        ]

        jobs: List[Optional[int]] = []
        adds: List[Tuple[Tensor, int, int]] = []   # (dst row indices, staging row, n)
        with torch.cuda.stream(stream), torch.no_grad():
            if own is not None:
                file_id, segs = own
                first, n, _ = segs[pid]
                assert n == self.part_sizes[pid]
                jobs.append(self._submit_read(
                    gpu_target, self._blob_row_runs(file_id, first, n, 0, row_bytes)
                ))
            else:
                gpu_target.zero_()

            staging = None
            if others:
                total = sum(n for _, (_, n, _) in others)
                staging = torch.empty(
                    (total, gpu_target.size(1)),
                    dtype=gpu_target.dtype,
                    device=gpu_target.device,
                )
                runs = []
                s = 0
                for file_id, (first, n, idx) in others:
                    runs.append(self._blob_row_runs(file_id, first, n, s, row_bytes))
                    adds.append((idx.to(gpu_target.device, torch.long), s, n))
                    s += n
                jobs.append(self._submit_read(staging, torch.cat(runs)))

        def post():
            if not adds:
                return
            with torch.cuda.stream(stream), torch.no_grad():
                for idx, s, n in adds:
                    gpu_target.index_add_(0, idx, staging[s:s + n])

        print(
            f"[read_grad] {self._file_prefix}_p{pid} own={'yes' if own else 'no'} "
            f"from_others={len(others)} rows={sum(n for _, (_, n, _) in others)}"
        )
        return PendingRead(self._ops, jobs, post)

    # ------------------------------------------------------------------
    # Scatter: one GPU tensor -> multiple host partitions (with accumulation)
    # ------------------------------------------------------------------

    def async_scatter(
        self,
        pid: int,
        gpu_source: Tensor,
        boundaries: List[Optional[Tensor]],
        stream: torch.cuda.Stream,
    ) -> None:
        """Scatter gradients from GPU to host partitions with accumulation.

        GPU layout matches gather: [intra(pid) | boundary_from_p0 | ...]
        Accumulates (+=) into existing host partition data.

        Args:
            pid: Source partition index.
            gpu_source: GPU tensor [total_size, dim].
            boundaries: Same format as gather.
            stream: CUDA stream.
        """
        assert gpu_source.is_cuda
        for src_pid, boundary in enumerate(boundaries):
            if src_pid == pid:
                self.allocate(src_pid)
                if not self._zero_initialized[src_pid]:
                    self._tensors[src_pid].zero_()
                    self._zero_initialized[src_pid] = True
            elif boundary is not None and boundary.numel() > 0:
                self.allocate(src_pid)
                if not self._zero_initialized[src_pid]:
                    self._tensors[src_pid].zero_()
                    self._zero_initialized[src_pid] = True

        bndries = []
        for i in range(self.num_parts):
            if i == pid or boundaries[i] is None:
                bndries.append(torch.empty(0, dtype=torch.long))
            else:
                bndries.append(boundaries[i])

        with torch.cuda.stream(stream):
            if self._ops is not None:
                self._ops.scatter_partitions(
                    pid, gpu_source, self._tensors, bndries
                )
            else:
                # Python fallback (synchronous)
                offset = self._tensors[pid].size(0)
                cpu_data = gpu_source[:offset].cpu()
                self._tensors[pid].add_(cpu_data)
                for i in range(self.num_parts):
                    if i == pid or bndries[i].numel() == 0:
                        continue
                    n = bndries[i].size(0)
                    cpu_slice = gpu_source[offset : offset + n].cpu()
                    self._tensors[i].index_put_(
                        (bndries[i],), cpu_slice, accumulate=True
                    )
                    offset += n

    # ------------------------------------------------------------------
    # Storage bypass: GPU -> NVMe directly
    # ------------------------------------------------------------------

    def async_bypass_to_storage(
        self, pid: int, gpu_data: Tensor, stream: torch.cuda.Stream
    ) -> None:
        """Bypass: write GPU tensor directly to NVMe storage.
        """
        assert self._backend is not None, "StorageBackend required for bypass"
        file_id = f"{self._file_prefix}_p{pid}"
        # The write starts in the background once `stream` reaches this point,
        # i.e. after the kernels that produced gpu_data. Call
        # wait_storage_writes(pid) before freeing or overwriting gpu_data.
        stream.wait_stream(torch.cuda.current_stream(gpu_data.device))
        self._backend.gpu_write(gpu_data, file_id, stream, async_=True)

    def wait_storage_writes(self, pid: Optional[int] = None) -> None:
        """Wait for async GPU->SSD writes of one partition (or all of this buffer)."""
        if self._backend is None:
            return
        if pid is None:
            self._backend.wait_gpu_writes(prefix=f"{self._file_prefix}_p")
        else:
            self._backend.wait_gpu_writes([f"{self._file_prefix}_p{pid}"])

    def async_bypass_to_storage_raw(
            self, pid: int, gpu_data: Tensor, stream: torch.cuda.Stream
        ) -> None:
            """Bypass: write GPU tensor directly to NVMe storage.
            """
            assert self._backend is not None, "StorageBackend required for bypass"
            file_id = f"{self._file_prefix}_p{pid}"
            stream.wait_stream(torch.cuda.current_stream(gpu_data.device))
            stream.synchronize()
            self._backend.host_write(gpu_data, file_id)

    # ------------------------------------------------------------------
    # Storage <-> Host (cache management)
    # ------------------------------------------------------------------

    def _ensure_allocated(self, pid: int) -> None:
        """Re-allocate host tensor if it was freed by cache eviction."""
        self.allocate(pid)

    def storage_to_cpu(self, phase = "none", pid: Optional[int] = None) -> None:
        """Load partition(s) from NVMe to host cache via io_uring.

        Re-allocates host tensor if previously freed by cache eviction.
        """
        assert self._backend is not None
        if pid is not None:
            self._ensure_allocated(pid)
            file_id = f"{self._file_prefix}_p{pid}"
            t0 = time.perf_counter_ns()
            h = self._backend.host_read(file_id, self._tensors[pid])
            self._backend.wait(h)
            tn = time.perf_counter_ns()

            stat.load_CPU_timestamp(phase, t0, tn)
            self._zero_initialized[pid] = True
        else:
            handles = []
            loaded = []
            for i in range(self.num_parts):
                file_id = f"{self._file_prefix}_p{i}"
                if self._backend.exists(file_id):
                    self._ensure_allocated(i)
                    h = self._backend.host_read(file_id, self._tensors[i])
                    handles.append(h)
                    loaded.append(i)
            for h in handles:
                self._backend.wait(h)
            for i in loaded:
                self._zero_initialized[i] = True


    def storage_to_gpu(self,
        phase,
        pid,
        gpu_target: Tensor,
        stream: torch.cuda.Stream) -> None:

        """Read partition pid from the SSD into gpu_target (FTL, same path as
        the forward gather). Wait with h2d_synchronize(stream, pid)."""
        assert self._backend is not None
        self.async_read_raw(pid, gpu_target, stream)


    def cpu_to_storage(self, pid: Optional[int] = None) -> None:
        """Flush partition(s) from host cache to NVMe via io_uring."""
        assert self._backend is not None
        if pid is not None:
            file_id = f"{self._file_prefix}_p{pid}"
            self.allocate(pid)
            h = self._backend.host_write(self._tensors[pid], file_id)
            #self._backend.wait(h)
        else:
            handles = []
            for i in range(self.num_parts):
                file_id = f"{self._file_prefix}_p{i}"
                self.allocate(i)
                h = self._backend.host_write(self._tensors[i], file_id)
                handles.append(h)
            for h in handles:
                self._backend.wait(h)



    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def initialize_zeros(self, lazy: bool = False) -> None:
        """Zero all partition tensors (for gradient write-back init).

        Re-allocates any tensors that were freed by cache eviction.
        """
        if lazy:
            self.release_all()
            return
        for pid in range(self.num_parts):
            self._ensure_allocated(pid)
            self._tensors[pid].zero_()
            self._zero_initialized[pid] = True

    def reset(self) -> None:
        """Reset all tensors to zero."""
        self.initialize_zeros()
