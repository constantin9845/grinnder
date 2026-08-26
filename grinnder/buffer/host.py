"""Host buffer: partition-wise tensor management on host RAM / NVMe storage.

Each HostBuffer manages one "layer" of activations or gradients, split into
per-partition tensors. Supports async gather, scatter, fill, upload, and
bypass-to-storage operations via C++ extensions.
"""

from __future__ import annotations

from typing import List, Optional

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
        return self._backend.exists(f"{self._file_prefix}_p{pid}")

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
        """
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


    def h2d_synchronize(self, stream: torch.cuda.Stream) -> None:
        """Wait for H2D operations to complete.

        Two-phase synchronization:
        1. C++ thread pool sync: wait for cudaMemcpyAsync calls to be submitted
        2. CUDA stream sync: wait for all submitted CUDA ops to complete
        """
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
            if self._ops is not None:
                self._ops.gather_partitions(pid, self._tensors, gpu_target, bndries)
                print("Not a fallback")
            else:
                # Python fallback
                print("Fallback")
                offset = self._tensors[pid].size(0)
                gpu_target[:offset].copy_(self._tensors[pid])
                for i in range(self.num_parts):
                    if i == pid or bndries[i].numel() == 0:
                        continue
                    selected = self._tensors[i].index_select(0, bndries[i])
                    print(selected)
                    exit(1)
                    n = selected.size(0)
                    gpu_target[offset : offset + n].copy_(selected)
                    offset += n
        
        tn = time.perf_counter_ns()
        stat.load_GPU_timestamp(phase, "copy", t0, tn)

        bt = gpu_target.numel() * gpu_target.element_size()
        gb = round(bt / (1024**3),3)

        print(f"\tPartition {pid} loads {gb} GB from other partitions")

    # ------------------------------------------------------------------
    # Gather Direct: target partition + boundary feat/act from NVMe -> one GPU tensor
    # ------------------------------------------------------------------

    def async_gather_direct(
    self,
    phase,
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

        tn = time.perf_counter_ns()
        stat.load_GPU_timestamp(phase, "gather", t0, tn)

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

        t0 = time.perf_counter_ns()

        with torch.cuda.stream(stream):

            # --------------------------------------------------------
            # Load full target partition
            # Equivalent to:
            #
            #   gpu_target[:offset].copy_(self._tensors[pid])
            # --------------------------------------------------------

            print("Fallback")

            offset = num_nodes[pid]

            print(f"\n--- [DEBUG Target Partition pid={pid}] ---")
            print(f"File: {file_paths[pid]}")
            print(
                f"File size: {file_sizes[pid]} bytes "
                f"({num_nodes[pid]} rows)"
            )
            print(
                f"Reading target slice: "
                f"gpu_target[0:{offset}] "
                f"(file_offset=0)"
            )

            self._backend.gpu_read(
                status=3,
                fd=None,
                file_id=file_paths[pid],
                tensor=gpu_target[:offset],
                file_offset=0,
                stream=stream,
            )

            print("Target partition loaded")

            # --------------------------------------------------------
            # Load only required boundary rows
            #
            # Equivalent to:
            #
            #   selected = self._tensors[i].index_select(0, bndries[i])
            #   gpu_target[offset : offset + n].copy_(selected)
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

                print(f"--- [DEBUG Boundary Partition src_pid={i}] ---")
                print(f"File: {part_file}")
                print(
                    f"File size: {part_file_size} bytes "
                    f"(Max valid node index: {max_valid_nodes - 1})"
                )
                print(f"Requested boundary index count: {num_rows}")
                print(
                    f"Indices min/max: "
                    f"min={indices.min().item()}, "
                    f"max={indices.max().item()}"
                )

                # These must be LOCAL row indices into partition i,
                # exactly as they were for self._tensors[i].index_select().
                if (
                    indices.min().item() < 0
                    or indices.max().item() >= max_valid_nodes
                ):
                    raise RuntimeError(
                        f"Boundary index out of range:\n"
                        f"  target pid       = {pid}\n"
                        f"  source pid       = {i}\n"
                        f"  file             = {part_file}\n"
                        f"  file size        = {part_file_size}\n"
                        f"  rows in file     = {max_valid_nodes}\n"
                        f"  requested rows   = {num_rows}\n"
                        f"  min index        = {indices.min().item()}\n"
                        f"  max index        = {indices.max().item()}\n"
                    )

                fd = None

                for k in range(num_rows):

                    node_idx = indices[k].item()

                    file_offset = node_idx * row_bytes

                    # Exactly one output row, preserving index_select order.
                    dest_row = gpu_target[
                        offset : offset + 1
                    ]

                    if file_offset + row_bytes > part_file_size:
                        raise RuntimeError(
                            f"\n[ERROR BOUNDS EXCEEDED]\n"
                            f"  target pid = {pid}\n"
                            f"  src_pid = {i}\n"
                            f"  k = {k}/{num_rows}\n"
                            f"  node_idx = {node_idx}\n"
                            f"  file_offset = {file_offset}\n"
                            f"  row_bytes = {row_bytes}\n"
                            f"  file end = {file_offset + row_bytes}\n"
                            f"  actual file size = {part_file_size}\n"
                            f"  file = {part_file}\n"
                        )

                    if k < 5 or k >= num_rows - 5:
                        print(
                            f"  [Read {k+1}/{num_rows}] "
                            f"node_idx={node_idx} -> "
                            f"file_offset={file_offset} "
                            f"(0x{file_offset:X}), "
                            f"gpu_target index={offset}, status="
                            f"{3 if num_rows == 1 else (0 if k == 0 else (2 if k == num_rows - 1 else 1))}"
                        )

                    if k == 0 and num_rows == 1:
                        # Single row: open, read, close
                        fd = self._backend.gpu_read(
                            status=3,
                            fd=None,
                            file_id=part_file,
                            tensor=dest_row,
                            file_offset=file_offset,
                            stream=stream,
                        )

                    elif k == 0:
                        # First row: open handle + read
                        fd = self._backend.gpu_read(
                            status=0,
                            fd=None,
                            file_id=part_file,
                            tensor=dest_row,
                            file_offset=file_offset,
                            stream=stream,
                        )

                    elif k != num_rows - 1:
                        # Middle row: use persistent fd
                        fd = self._backend.gpu_read(
                            status=1,
                            fd=fd,
                            file_id=part_file,
                            tensor=dest_row,
                            file_offset=file_offset,
                            stream=stream,
                        )

                    else:
                        # Last row: read + close
                        fd = self._backend.gpu_read(
                            status=2,
                            fd=fd,
                            file_id=part_file,
                            tensor=dest_row,
                            file_offset=file_offset,
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

        tn = time.perf_counter_ns()
        stat.load_GPU_timestamp(phase, "copy", t0, tn)

        bt = gpu_target.numel() * gpu_target.element_size()
        gb = round(bt / (1024**3), 3)

        print(
            f"\tPartition {pid} loads "
            f"{gb} GB from other partitions"
        )

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

        With GDS, kvikio uses a direct GPU-to-NVMe path. If direct GDS is not
        available, kvikio may use a compatible staging path internally.
        """
        assert self._backend is not None, "StorageBackend required for bypass"
        file_id = f"{self._file_prefix}_p{pid}"
        stream.wait_stream(torch.cuda.current_stream(gpu_data.device))
        stream.synchronize()
        self._backend.gpu_write(gpu_data, file_id, stream)

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

    def cpu_to_storage(self, pid: Optional[int] = None) -> None:
        """Flush partition(s) from host cache to NVMe via io_uring."""
        assert self._backend is not None
        if pid is not None:
            file_id = f"{self._file_prefix}_p{pid}"
            self.allocate(pid)
            h = self._backend.host_write(self._tensors[pid], file_id)
            self._backend.wait(h)
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
