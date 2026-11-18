"""Storage backend with dual I/O engines.

GPU <-> Storage: kvikio (GPUDirect Storage)
Host <-> Storage: io_uring (bundled liburing 2.8)

Both engines are REQUIRED — no POSIX fallbacks.
"""

from __future__ import annotations

import contextlib
import os
import shutil
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import Tensor

from grinnder.utils import ensure_dir
from grinnder.stats import stat, timed, timed_by, trace
from grinnder.ftl import FTL


class StorageBackend:
    """Manages NVMe file I/O with two dedicated engines.

    - GPU <-> Storage: kvikio (GPUDirect Storage) for bypass writes and
      backward activation loads. Direct GPU-NVMe transfers.
    - Host <-> Storage: io_uring (C++, bundled liburing 2.8) for cache
      loads/flushes and gradient flush. Kernel-level async I/O.

    All files stored under ``storage_dir/{file_id}.pt``.
    """

    def __init__(
        self,
        storage_dir: str,
        queue_depth: int = 64,
        io_mode: str = "ours",
        gds: bool = False,
    ):
        assert io_mode in ("ours", "base"), f"unknown io_mode {io_mode!r}"
        self.io_mode = io_mode
        # Ours only: cuFile batch I/O (True) or NVMe passthrough + pinned
        # host buffer (False). Base never uses either.
        if gds and io_mode != "ours":
            print("[storage] gds only applies to io_mode=ours; ignored")
        self.gds = bool(gds) and io_mode == "ours"
        if io_mode == "ours":
            self._set_gds_mode(self.gds)
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
        self.host_ftl = FTL(gds=self.gds) if io_mode == "ours" else None
        if io_mode == "ours":
            print("[storage] ours I/O path: "
                  + ("GDS (cuFile batch)" if self.gds
                     else "NVMe passthrough + pinned host buffer"))

        # file_id -> job id of an async GPU write still in flight (ours only)
        # file_id -> (write job, breakdown key of waiting for it)
        self._pending_gpu_writes: Dict[str, Tuple[int, str]] = {}
        # base mode: io_uring handle -> breakdown key of waiting for it
        self._handle_keys: Dict[int, str] = {}
        # base: pinned staging buffers for GPU -> file writes (allocated on first use)
        self._staging: Optional[list] = None

    @staticmethod
    def _set_gds_mode(gds: bool) -> None:
        """Pass the GDS choice (config.gds / --gds) to the C++ side, which
        applies the real-GDS workarounds when it is on. This is the only
        switch; GRINNDER_GDS_DIRECT is no longer read."""
        if os.environ.get("GRINNDER_GDS_DIRECT"):
            print("[storage] GRINNDER_GDS_DIRECT is ignored; use --gds 0|1")
        try:
            from grinnder._C import set_gds_direct, get_gds_direct
        except ImportError as e:
            raise ImportError(
                "grinnder._C has no set_gds_direct: rebuild the C++ extension "
                "(pip install -e . --no-build-isolation)"
            ) from e
        set_gds_direct(bool(gds))
        assert get_gds_direct() == bool(gds)

    @property
    def uses_ftl(self) -> bool:
        return self.host_ftl is not None

    def _path(self, file_id: str) -> str:
        return os.path.join(self._storage_dir, f"{file_id}.pt")

    # ------------------------------------------------------------------
    # GPU <-> Storage (kvikio / GPUDirect Storage)
    # ------------------------------------------------------------------

    @timed_by(lambda self, *a, kind="activation", **k: f"ssd.write.{kind}.submit")
    def gpu_write(
        self,
        tensor: Tensor,
        file_id: str,
        stream: Optional[torch.cuda.Stream] = None,
        async_: bool = False,
        kind: str = "activation",
    ) -> None:
        """Write a GPU tensor directly to NVMe via GDS (host FTL LBA runs).

        Detaches from autograd graph — bypass writes data only.
        Activations are rewritten every epoch: a tensor already mapped with
        the same size keeps its mapping and is overwritten in place.

        With async_=True the write runs in the background (cuFile batch API).
        Call wait_gpu_writes() before freeing or overwriting `tensor`;
        reads of the same file_id wait automatically.

        kind ("activation" | "gradient") only labels the time breakdown.
        """
        assert kind in ("activation", "gradient"), kind

        assert tensor.is_cuda, "gpu_write requires a CUDA tensor"
        assert tensor.is_contiguous(), "gpu_write requires contiguous tensor"

        if not self.uses_ftl:
            # base (original GriNNder): whole tensor to its file, synchronous.
            self._base_gpu_write(tensor, file_id, stream, kind)
            return

        # A previous write of this file must land before we overwrite it.
        self.wait_gpu_writes([file_id])

        ftl = self.host_ftl
        nbytes = tensor.numel() * tensor.element_size()
        meta = ftl.tensor_table.get(file_id)
        if meta is not None and meta["total_bytes"] != nbytes:
            with stat.timed("ftl.map.unmap"):
                ftl.delete_tensor(file_id)
            stat.breakdown.count("ftl.map.unmap")
            meta = None
        if meta is None:
            with stat.timed("ftl.map.alloc"):
                meta = ftl.map_tensor(file_id, tensor)
            stat.breakdown.count("ftl.map.alloc")
            stat.breakdown.count("ftl.pages.alloc", meta["num_pages"])

        # The write starts once all work queued so far on `stream`
        # (or the current stream) has finished.
        with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
            job = ftl.gpu_write(meta, tensor, file_id, async_=async_)
        if trace.recording and async_ and job is not None:
            trace.event("job", job=job, what=f"write.{kind}", file=file_id, bytes=nbytes)
        if async_:
            self._pending_gpu_writes[file_id] = (job, f"ssd.write.{kind}.wait")

    # GPU -> file in base mode goes through two reusable pinned buffers of
    # this size, so host memory use stays fixed (and O_DIRECT keeps the data
    # out of the page cache).
    _BASE_STAGING_BYTES = 64 << 20

    def _base_gpu_write(
        self,
        tensor: Tensor,
        file_id: str,
        stream: Optional[torch.cuda.Stream],
        kind: str = "activation",
    ) -> None:
        wait_key = f"ssd.write.{kind}.wait"
        stream = stream or torch.cuda.current_stream(tensor.device)
        stream.synchronize()
        src = tensor.detach().contiguous().view(-1).view(torch.uint8)
        nbytes = src.numel()
        path = self._path(file_id)
        if nbytes == 0:
            self.wait(self._io_engine.submit_write(path, src.cpu(), 0, 0), key=wait_key)
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
                self.wait(pending[slot], key=wait_key)  # staging buffer free again
            n = min(chunk, nbytes - off)
            buf = self._staging[slot]
            with stat.timed("transfer.staging"):  # GPU -> pinned staging
                with torch.cuda.stream(stream):
                    buf[:n].copy_(src[off:off + n], non_blocking=True)
                stream.synchronize()
            # Chunks start at multiples of 64 MiB, so file offsets stay aligned.
            pending[slot] = self._io_engine.submit_write(path, buf, off, n)
        for h in pending:
            self.wait(h, key=wait_key)

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
            job, key = self._pending_gpu_writes.pop(f)
            with stat.timed(key):
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

    @timed("ssd.host.write")
    def host_write(self, tensor: Tensor, file_id: str, async_: bool = True) -> Optional[int]:
        """Write a CPU tensor to storage.

        ours: FTL device write (synchronous, verified); returns None.
        base: io_uring write to the tensor's file (original); returns a
              handle for wait().
        """
        assert not tensor.is_cuda, "host_write requires a CPU tensor"
        assert tensor.is_contiguous(), "host_write requires contiguous tensor"

        if not self.uses_ftl:
            h = self._io_engine.submit_write(
                self._path(file_id), tensor, 0, tensor.numel() * tensor.element_size()
            )
            self._handle_keys[h] = "ssd.host.write"
            return h

        with stat.timed("ftl.map.alloc"):
            meta = self.host_ftl.map_tensor(file_id, tensor)
        stat.breakdown.count("ftl.map.alloc")
        stat.breakdown.count("ftl.pages.alloc", meta["num_pages"])
        self.host_ftl.device_write(meta, tensor, file_id)

        with stat.timed("ftl.verify"):
            verify = self.host_ftl.verify_write(meta, tensor, file_id)
        assert verify, f"Data verification failed for file_id={file_id}! DRAM contents corrupted."
        return None


    @timed("ssd.host.read")
    def host_read(self, file_id: str, tensor: Tensor, async_: bool = True) -> int:
        """Read a file into a CPU tensor via io_uring (base mode).

        Returns:
            Handle for wait().
        """
        assert not self.uses_ftl, "host_read reads files (io_mode=base); ours reads via the FTL"
        path = self._path(file_id)
        assert not tensor.is_cuda, "host_read requires a CPU tensor"
        assert tensor.is_contiguous(), "host_read requires contiguous tensor"

        h = self._io_engine.submit_read(
            path, tensor, 0, tensor.numel() * tensor.element_size()
        )
        self._handle_keys[h] = "ssd.host.read"
        return h

    def wait(self, handle: Optional[int], key: Optional[str] = None) -> None:
        """Wait for an async io_uring operation to complete (None: nothing to wait).

        The time counts for `key`, else for the key host_read/host_write
        recorded for the handle.
        """
        if handle is None:
            return
        key = key or self._handle_keys.pop(handle, "ssd")
        self._handle_keys.pop(handle, None)
        with stat.timed(key):
            self._io_engine.wait(handle)

    @timed("ssd")
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
