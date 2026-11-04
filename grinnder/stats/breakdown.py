"""Where the training time goes: SSD access vs data movement vs compute.

Each epoch's wall-clock time is split by what the training loop is waiting on
at that moment (the exposed time). Work that overlaps with something else,
e.g. a prefetch running during compute, costs nothing here; only waiting for
it does. The categories add up to the epoch time:

  ssd       waiting for SSD reads/writes (io_uring O_DIRECT files in base,
            FTL / GPU direct reads and writes in ours)
  transfer  data movement between host and GPU: H2D/D2H copies, CPU-side
            gather/scatter of boundary rows, adjacency uploads
  compute   GPU forward/backward/loss kernels and the optimizer step
  other     everything else (Python, cache bookkeeping, logging)

Timers nest: time inside an inner timer counts only for the inner category,
e.g. an SSD load inside a base-mode gather counts as ssd, not transfer.
The stack is shared across threads because backward runs on PyTorch's
autograd thread while the main thread waits inside .backward().
"""

from __future__ import annotations

import json
import statistics
import threading
import time
from typing import Dict, List, Optional

CATEGORIES = ("ssd", "transfer", "compute")
LABELS = {
    "ssd": "SSD access",
    "transfer": "data movement",
    "compute": "compute",
    "other": "other",
}
PHASES = ("forward", "loss", "backward", "optimizer")


class _Frame:
    __slots__ = ("category", "t0", "child_ns")

    def __init__(self, category: str, t0: int):
        self.category = category
        self.t0 = t0
        self.child_ns = 0


class _Timer:
    __slots__ = ("_bd", "_category", "_frame")

    def __init__(self, bd: "TimeBreakdown", category: str):
        self._bd = bd
        self._category = category
        self._frame: Optional[_Frame] = None

    def __enter__(self):
        if self._bd.active:
            self._frame = self._bd._push(self._category)
        return self

    def __exit__(self, *exc):
        if self._frame is not None:
            self._bd._pop(self._frame)
            self._frame = None
        return False


class TimeBreakdown:
    def __init__(self):
        self.active = False
        self.epochs: List[dict] = []
        self._lock = threading.Lock()
        self._stack: List[_Frame] = []
        self._phase: Optional[str] = None
        self._phase_t0 = 0
        self._epoch_t0 = 0
        self._ns: Dict[str, Dict[str, int]] = {}
        self._wall_ns: Dict[str, int] = {}

    # ---- recording -----------------------------------------------------

    def timed(self, category: str) -> _Timer:
        """Context manager: time spent inside counts for `category`."""
        assert category in CATEGORIES, category
        return _Timer(self, category)

    def begin_epoch(self) -> None:
        self._stack = []
        self._ns = {p: {c: 0 for c in CATEGORIES} for p in PHASES}
        self._wall_ns = {p: 0 for p in PHASES}
        self._phase = None
        self._epoch_t0 = self._phase_t0 = time.perf_counter_ns()
        self.active = True

    def set_phase(self, phase: str) -> None:
        assert phase in PHASES, phase
        now = time.perf_counter_ns()
        self._close_phase(now)
        self._phase = phase
        self._phase_t0 = now

    def end_epoch(self) -> dict:
        now = time.perf_counter_ns()
        self._close_phase(now)
        self.active = False
        record = {"wall_s": (now - self._epoch_t0) / 1e9, "phases": {}}
        for p in PHASES:
            cats = {c: self._ns[p][c] / 1e9 for c in CATEGORIES}
            wall = self._wall_ns[p] / 1e9
            cats["other"] = max(0.0, wall - sum(cats.values()))
            record["phases"][p] = {"wall_s": wall, **cats}
        # Setup outside the phases (buffer reset etc.) counts as other.
        phase_wall = sum(self._wall_ns.values()) / 1e9
        record["outside_phases_s"] = max(0.0, record["wall_s"] - phase_wall)
        self.epochs.append(record)
        return record

    def _close_phase(self, now: int) -> None:
        if self._phase is not None:
            self._wall_ns[self._phase] += now - self._phase_t0

    def _push(self, category: str) -> _Frame:
        frame = _Frame(category, time.perf_counter_ns())
        with self._lock:
            self._stack.append(frame)
        return frame

    def _pop(self, frame: _Frame) -> None:
        now = time.perf_counter_ns()
        total = now - frame.t0
        with self._lock:
            parent = None
            # Normally the top of the stack; search in case threads interleaved.
            for i in range(len(self._stack) - 1, -1, -1):
                if self._stack[i] is frame:
                    del self._stack[i]
                    if i > 0:
                        parent = self._stack[i - 1]
                    break
            if self._phase is not None:
                self._ns[self._phase][frame.category] += max(0, total - frame.child_ns)
            if parent is not None:
                parent.child_ns += total

    # ---- reporting -----------------------------------------------------

    @staticmethod
    def _totals(record: dict) -> Dict[str, float]:
        out = {c: 0.0 for c in (*CATEGORIES, "other")}
        for p in PHASES:
            for c in out:
                out[c] += record["phases"][p][c]
        out["other"] += record["outside_phases_s"]
        return out

    def print_epoch(self, record: Optional[dict] = None, title: str = "") -> None:
        record = record or self.epochs[-1]
        cols = (*CATEGORIES, "other")
        w = 17
        print(f"\n========== TIME BREAKDOWN {title} (exposed time) ==========")
        print(f"{'phase':<10}" + "".join(f"{LABELS[c]:>{w}}" for c in cols) + f"{'total':>{w - 5}}")

        def row(name: str, vals: Dict[str, float], wall: float) -> None:
            cells = "".join(
                f"{vals[c]:>{w - 7}.2f}s {100 * vals[c] / wall if wall else 0:>3.0f}%"
                for c in cols
            )
            print(f"{name:<10}{cells}{wall:>{w - 6}.2f}s")

        for p in PHASES:
            ph = record["phases"][p]
            row(p, ph, ph["wall_s"])
        row("TOTAL", self._totals(record), record["wall_s"])

    def print_summary(self, skip_first: bool = True) -> None:
        """Median over epochs (the first one skipped when there are more,
        since it includes warm-up)."""
        epochs = self.epochs[1:] if skip_first and len(self.epochs) > 1 else self.epochs
        if not epochs:
            return
        med = lambda xs: statistics.median(xs)
        summary = {
            "wall_s": med([e["wall_s"] for e in epochs]),
            "outside_phases_s": med([e["outside_phases_s"] for e in epochs]),
            "phases": {
                p: {
                    k: med([e["phases"][p][k] for e in epochs])
                    for k in ("wall_s", *CATEGORIES, "other")
                }
                for p in PHASES
            },
        }
        n = len(epochs)
        self.print_epoch(summary, title=f"median of {n} epoch{'s' if n > 1 else ''}")

    def save(self, path: str) -> None:
        """All epochs as JSON (seconds), for plots and tables."""
        with open(path, "w") as f:
            json.dump(
                {
                    "categories": {c: LABELS[c] for c in (*CATEGORIES, "other")},
                    "epochs": [
                        {**e, "totals": self._totals(e)} for e in self.epochs
                    ],
                },
                f,
                indent=2,
            )
