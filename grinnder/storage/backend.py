"""Storage backend with dual I/O engines.

GPU <-> Storage: kvikio (GPUDirect Storage)
Host <-> Storage: io_uring (bundled liburing 2.8)

Both engines are REQUIRED — no POSIX fallbacks.
"""

from __future__ import annotations

import contextlib
import os
import shutil
from typing import Dict, Iterable, Optional

import torch
from torch import Tensor

from grinnder.utils import ensure_dir
from grinnder.stats import stat
from grinnder.ftl import FTL


class StorageBackend:
    """Manages NVMe file I/O with two dedicated engines.

    - GPU <-> Storage: kvikio (GPUDirect Storage) for bypass writes and
      backward activation loads. Direct GPU-NVMe transfers.
    - Host <-> Storage: io_uring (C++, bundled liburing 2.8) for cache
      loads/flushes and gradient flush. Kernel-level async I/O.

    All files stored under ``storage_dir/{file_id}.pt``.
    """

    def __init__(self, storage_dir: str, queue_depth: int = 64, io_mode: str = "ours"):
        assert io_mode in ("ours", "base"), f"unknown io_mode {io_mode!r}"
        self.io_mode = io_mode
        self._storage_dir = ensure_dir(storage_dir)

        # GPU <-> Storage: kvikio required
        try:
            import kvikio
            self._kvikio = kvikio
        except ImportError:
            raise ImportError(
                "kvikio is required for GriNNder's grinnder mode. "
                "Install with: pip install kvikio-cu12"
            )

        # Host <-> Storage: io_uring required (bundled C++ extension)
        try:
            from grinnder._C import IoUringEngine
            self._io_engine = IoUringEngine(queue_depth)
        except ImportError:
            raise ImportError(
                "grinnder._C not found. Build C++ extensions with: "
                "pip install -e . --no-build-isolation"
            )
        if not self._io_engine.has_io_uring():
            raise RuntimeError(
                "io_uring initialization failed. Ensure liburing is available "
                "and the kernel supports io_uring (Linux 5.1+). "
                "The bundled liburing is in third_party/liburing/."
            )
        
        # ours: host FTL on the raw device. base: plain files under storage_dir,
        # the raw device is never touched.
        self.host_ftl = FTL() if io_mode == "ours" else None

        # file_id -> job id of an async GPU write still in flight (ours only)
        self._pending_gpu_writes: Dict[str, int] = {}
        # base: pinned staging buffers for GPU -> file writes (allocated on first use)
        self._staging: Optional[list] = None

    @property
    def uses_ftl(self) -> bool:
        return self.host_ftl is not None

    def _path(self, file_id: str) -> str:
        return os.path.join(self._storage_dir, f"{file_id}.pt")

    # ------------------------------------------------------------------
    # GPU <-> Storage (kvikio / GPUDirect Storage)
    # ------------------------------------------------------------------

    def gpu_write(
        self,
        tensor: Tensor,
        file_id: str,
        stream: Optional[torch.cuda.Stream] = None,
        async_: bool = False,
    ) -> None:
        """Write a GPU tensor directly to NVMe via GDS (host FTL LBA runs).

        Detaches from autograd graph — bypass writes data only.
        Activations are rewritten every epoch: a tensor already mapped with
        the same size keeps its mapping and is overwritten in place.

        With async_=True the write runs in the background (cuFile batch API).
        Call wait_gpu_writes() before freeing or overwriting `tensor`;
        reads of the same file_id wait automatically.
        """

        assert tensor.is_cuda, "gpu_write requires a CUDA tensor"
        assert tensor.is_contiguous(), "gpu_write requires contiguous tensor"

        if not self.uses_ftl:
            # base (original GriNNder): whole tensor to its file, synchronous.
            self._base_gpu_write(tensor, file_id, stream)
            return

        # A previous write of this file must land before we overwrite it.
        self.wait_gpu_writes([file_id])

        ftl = self.host_ftl
        nbytes = tensor.numel() * tensor.element_size()
        meta = ftl.tensor_table.get(file_id)
        if meta is not None and meta["total_bytes"] != nbytes:
            ftl.delete_tensor(file_id)
            meta = None
        if meta is None:
            meta = ftl.map_tensor(file_id, tensor)

        # The write starts once all work queued so far on `stream`
        # (or the current stream) has finished.
        with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
            job = ftl.gpu_write(meta, tensor, file_id, async_=async_)
        if async_:
            self._pending_gpu_writes[file_id] = job

    # GPU -> file in base mode goes through two reusable pinned buffers of
    # this size, so host memory use stays fixed (and O_DIRECT keeps the data
    # out of the page cache).
    _BASE_STAGING_BYTES = 64 << 20

    def _base_gpu_write(
        self, tensor: Tensor, file_id: str, stream: Optional[torch.cuda.Stream]
    ) -> None:
        stream = stream or torch.cuda.current_stream(tensor.device)
        stream.synchronize()
        src = tensor.detach().contiguous().view(-1).view(torch.uint8)
        nbytes = src.numel()
        path = self._path(file_id)
        if nbytes == 0:
            self.wait(self._io_engine.submit_write(path, src.cpu(), 0, 0))
            return

        if self._staging is None:
            # Pinned allocations are page aligned, as O_DIRECT requires.
            self._staging = [
                torch.empty(self._BASE_STAGING_BYTES, dtype=torch.uint8, pin_memory=True)
                for _ in range(2)
            ]
        chunk = self._BASE_STAGING_BYTES
        pending: list = [None, None]
        for i, off in enumerate(range(0, nbytes, chunk)):
            slot = i % 2
            if pending[slot] is not None:
                self.wait(pending[slot])  # staging buffer free again
            n = min(chunk, nbytes - off)
            buf = self._staging[slot]
            with torch.cuda.stream(stream):
                buf[:n].copy_(src[off:off + n], non_blocking=True)
            stream.synchronize()
            # Chunks start at multiples of 64 MiB, so file offsets stay aligned.
            pending[slot] = self._io_engine.submit_write(path, buf, off, n)
        for h in pending:
            self.wait(h)

    def wait_gpu_writes(
        self,
        file_ids: Optional[Iterable[str]] = None,
        prefix: Optional[str] = None,
    ) -> None:
        """Wait for async GPU writes.

        file_ids: wait for these files; prefix: wait for files starting with it;
        neither: wait for all pending writes.
        """
        if not self._pending_gpu_writes:
            return
        if file_ids is None and prefix is None:
            targets = list(self._pending_gpu_writes)
        elif file_ids is not None:
            targets = [f for f in file_ids if f in self._pending_gpu_writes]
        else:
            targets = [f for f in self._pending_gpu_writes if f.startswith(prefix)]
        for f in targets:
            job = self._pending_gpu_writes.pop(f)
            self.host_ftl.gpu_write_wait(job)

        #verify = self.host_ftl.verify_write(meta, tensor, file_id)
        #assert verify, f"Data verification failed for file_id={file_id}! DRAM contents corrupted."
        return 0
        

    def gpu_read(
        self,
        file_id: str,
        tensor: Tensor,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """Read from NVMe directly into GPU tensor via GDS."""
        path = self._path(file_id)
        assert tensor.is_cuda, "gpu_read requires a CUDA tensor"
        assert tensor.is_contiguous(), "gpu_read requires contiguous tensor"

        f = self._kvikio.CuFile(path, "r")
        f.read(tensor, file_offset=0)
        f.close()

        stat.load_GDS_timestamp()

    def gpu_read_direct(
            self,
            status: int,
            fd,
            file_id: str,
            tensor: Tensor,
            offset: int,
            stream: Optional[torch.cuda.Stream] = None,
        ) -> None:
            """Read from NVMe directly into GPU tensor via GDS."""
            assert tensor.is_cuda, "gpu_read requires a CUDA tensor"
            tensor_detached = tensor.detach()
            assert tensor_detached.is_contiguous(), "gpu_read requires contiguous tensor"
    
            # Single read cycle (full file, etc) : open - read - close
            if status == 0:
                path = self._path(file_id)
                f = self._kvikio.CuFile(path, "r")
                f.read(tensor_detached, file_offset=offset)
                f.close()
                return None
    
            # First read : open - read
            elif status == 1:
                path = self._path(file_id)
                f = self._kvikio.CuFile(path, "r")
                f.read(tensor_detached, file_offset=offset)
                return f
    
            # Intermediate read : read
            elif status == 2:
                fd.read(tensor_detached, file_offset=offset)
                return fd
    
            # last read : read - close
            elif status == 3:
                fd.read(tensor_detached, file_offset=offset)
                fd.close()
                return None
    
            else: 
                print("Unknown gpu read status")
                exit(1)
    
            stat.load_GDS_timestamp()
    

    # --------------------------------------------------------------------------
    # Host <-> Storage : used for initial storage of partitioned graph features
    # --------------------------------------------------------------------------

    def host_write(self, tensor: Tensor, file_id: str, async_: bool = True) -> Optional[int]:
        """Write a CPU tensor to storage.

        ours: FTL device write (synchronous, verified); returns None.
        base: io_uring write to the tensor's file (original); returns a
              handle for wait().
        """
        assert not tensor.is_cuda, "host_write requires a CPU tensor"
        assert tensor.is_contiguous(), "host_write requires contiguous tensor"

        if not self.uses_ftl:
            return self._io_engine.submit_write(
                self._path(file_id), tensor, 0, tensor.numel() * tensor.element_size()
            )

        meta = self.host_ftl.map_tensor(file_id, tensor)
        self.host_ftl.device_write(meta, tensor, file_id)

        verify = self.host_ftl.verify_write(meta, tensor, file_id)
        assert verify, f"Data verification failed for file_id={file_id}! DRAM contents corrupted."
        return None


    def host_read(self, file_id: str, tensor: Tensor, async_: bool = True) -> int:
        """Read a file into a CPU tensor via io_uring (base mode).

        Returns:
            Handle for wait().
        """
        assert not self.uses_ftl, "host_read reads files (io_mode=base); ours reads via the FTL"
        path = self._path(file_id)
        assert not tensor.is_cuda, "host_read requires a CPU tensor"
        assert tensor.is_contiguous(), "host_read requires contiguous tensor"

        return self._io_engine.submit_read(
            path, tensor, 0, tensor.numel() * tensor.element_size()
        )

    def wait(self, handle: Optional[int]) -> None:
        """Wait for an async io_uring operation to complete (None: nothing to wait)."""
        if handle is not None:
            self._io_engine.wait(handle)

    def wait_all(self) -> None:
        """Wait for all pending io_uring operations."""
        self._io_engine.wait_all()

    # ------------------------------------------------------------------
    # File management
    # ------------------------------------------------------------------

    def exists(self, file_id: str) -> bool:
        """ours: the FTL has a mapping for it. base: its file exists."""
        if self.uses_ftl:
            return file_id in self.host_ftl.tensor_table
        return os.path.exists(self._path(file_id))

    def remove(self, file_id: str) -> None:
        path = self._path(file_id)
        if os.path.exists(path):
            os.remove(path)

    def cleanup(self) -> None:
        """Remove all files in storage directory."""
        if os.path.exists(self._storage_dir):
            shutil.rmtree(self._storage_dir)
            ensure_dir(self._storage_dir)

    def file_size(self, file_id: str) -> int:
        path = self._path(file_id)
        return os.path.getsize(path) if os.path.exists(path) else 0
