"""Chronological event trace of a training epoch (--trace).

Python events (this module) and the GDS worker threads' events (C++,
async_transfer.cu) go into one buffer in the C++ extension. Every event gets a
sequence number and a CLOCK_MONOTONIC timestamp under one lock, so `seq` is
the exact order events were recorded in, across all threads.

One JSON line per event, one file per epoch:
    {"seq": 812, "t_ns": 5821..., "tid": 4411, "ev": "gds.io_done",
     "args": {"job": 17, "batch": 0, "io": 42, "dev_off": ..., "bytes": 131072, "ok": 1}}

Events
  Python (main / autograd thread):
    epoch.begin / epoch.end   epoch
    phase                     phase (forward | loss | backward | optimizer)
    begin / end               key: every TimeBreakdown timer (ftl.*, ssd.*, compute.*, ...)
    partition                 phase, layer, pid: start of one partition's step
    job                       job id + what it is for (gather / partition / gradient / write, pid, ...)
  C++, GDS path only (--gds 1):
    gds.submit       caller   job, op (read|write), runs, ios, bytes
    gds.job_start    worker   job, ready_wait_ns (GPU work the job had to wait for)
    gds.ios          worker   job, ios, bytes (writes: I/O list built on the worker)
    gds.lock         worker   job, wait_ns (wait for other jobs' batches, global batch lock)
    gds.batch_submit worker   job, batch, first_io, n
    gds.io_done      worker   job, batch, io, dev_off, bytes, ok (completion order)
    gds.batch_done   worker   job, batch, done
    gds.tail_copy    worker   job, n (padded last page: scratch -> dst)
    gds.job_end      worker   job, ok
    gds.wait_begin / gds.wait_end   caller blocked on job

Only the GDS path has I/O events; with --gds 0 the trace has the Python
events plus gds.wait_begin / gds.wait_end (the NVMe jobs share the wait).
"""

from __future__ import annotations

import json
import os
import threading
from typing import Optional

# Drain the C++ buffer into the file once it holds this many events, so a
# long epoch does not keep everything in memory.
_FLUSH_EVERY = 200_000


class Trace:
    def __init__(self):
        self.enabled = False      # --trace given
        self.recording = False    # inside an epoch
        self._ops = None
        self._pattern: Optional[str] = None
        self._file = None
        self._epoch = 0           # first epoch is 1, as the training scripts number them
        self._n_since_check = 0
        self.paths = []
        self._flush_lock = threading.Lock()   # main and autograd threads both log

    def start(self, pattern: str) -> None:
        """Enable tracing; epoch i is written to pattern.format(i)."""
        try:
            import grinnder._C as ops
        except ImportError as e:
            raise RuntimeError("--trace needs the grinnder C++ extension") from e
        if not hasattr(ops, "trace_drain"):
            raise RuntimeError(
                "--trace: the C++ extension is older than this code; rebuild it "
                "with `pip install -e . --no-build-isolation`"
            )
        self._ops = ops
        self._pattern = pattern
        self.enabled = True

    # ---- epochs -----------------------------------------------------------

    def begin_epoch(self) -> None:
        if not self.enabled:
            return
        self._epoch += 1
        path = self._pattern.format(self._epoch)
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        self._ops.trace_drain()   # drop anything recorded between epochs
        self._file = open(path, "w")
        self.paths.append(path)
        self.recording = True
        self._ops.trace_enable(True)
        self.event("epoch.begin", epoch=self._epoch)

    def end_epoch(self) -> Optional[str]:
        """Write the rest of the epoch and stop recording. Events of I/O jobs
        still running after this (writes nobody waited for) are not recorded."""
        if not self.recording:
            return None
        self.event("epoch.end", epoch=self._epoch)
        self._ops.trace_enable(False)
        self.recording = False
        self.flush()
        self._file.close()
        self._file = None
        return self.paths[-1]

    # ---- events -----------------------------------------------------------

    def event(self, name: str, **args) -> None:
        if not self.recording:
            return
        self._ops.trace_event(name, json.dumps(args, default=str) if args else "{}")
        self._n_since_check += 1
        if self._n_since_check >= 1024:
            self._n_since_check = 0
            if self._ops.trace_pending() >= _FLUSH_EVERY:
                self.flush()

    def flush(self) -> None:
        # Drain and write under one lock, so the file stays in seq order.
        with self._flush_lock:
            if self._file is None:
                return
            write = self._file.write
            for seq, t_ns, tid, name, args in self._ops.trace_drain():
                write(f'{{"seq":{seq},"t_ns":{t_ns},"tid":{tid},"ev":"{name}","args":{args}}}\n')


trace = Trace()
