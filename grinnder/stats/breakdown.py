"""Where the training time goes: SSD access, FTL, data movement, compute.

Each epoch's wall-clock time is split by what the training loop is waiting on
at that moment (the exposed time). Work that overlaps with something else,
e.g. a prefetch running during compute, costs nothing here; only waiting for
it does. The top-level categories add up to the epoch time:

  ssd       waiting for SSD reads/writes (io_uring O_DIRECT files in base,
            FTL / GPU direct reads and writes in ours)
  ftl       host-side FTL work in ours: mapping, L2P lookups, row -> device
            offset translation, building and merging I/O runs
  transfer  data movement between host and GPU: H2D/D2H copies, CPU-side
            gather/scatter of boundary rows, adjacency uploads
  compute   GPU forward/backward/loss kernels and the optimizer step
  other     everything else (Python, cache bookkeeping, logging)

Every category has a tree of sub-keys (TREE), timed with dotted keys such as
timed("ssd.read.gather.wait"). Reports roll leaves up to every level. A bare
top-level key (timed("ssd")) still works; it shows as "ssd.(unattributed)".

Timers nest: time inside an inner timer counts only for the inner key,
e.g. an FTL lookup inside an SSD read counts as ftl, not ssd.
The stack is shared across threads because backward runs on PyTorch's
autograd thread while the main thread waits inside .backward().

Besides time, count() records per-phase counters (events, rows, bytes) and
snapshot() stores end-of-epoch state (e.g. ftl.stats()).
"""

from __future__ import annotations

import json
import statistics
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional

# Leaves per top-level category, in report order.
TREE: Dict[str, tuple] = {
    "ssd": (
        "read.gather.submit", "read.gather.wait",
        "read.partition.submit", "read.partition.wait",
        "read.gradient.submit", "read.gradient.wait",
        "write.activation.submit", "write.activation.wait",
        "write.gradient.submit", "write.gradient.wait",
        "host.read", "host.write",
    ),
    "ftl": (
        "map.alloc", "map.unmap",
        "lookup.walk",
        "translate.rows",
        "runs.build", "runs.merge",
        "verify", "log",
    ),
    "transfer": (
        "h2d.features", "h2d.gradient", "h2d.adjacency", "h2d.labels",
        "d2h.activation", "d2h.gradient",
        "staging",
        "cpu.gather", "cpu.scatter",
        "checkpoint",
    ),
    "compute": (
        "forward", "recompute", "backward", "loss", "grad_sum", "optimizer",
    ),
}
CATEGORIES = tuple(TREE)
ALL_CATEGORIES = (*CATEGORIES, "other")
UNATTRIBUTED = "(unattributed)"
KEYS = frozenset(CATEGORIES) | {f"{c}.{leaf}" for c, leaves in TREE.items() for leaf in leaves}
LABELS = {
    "ssd": "SSD access",
    "ftl": "FTL (host)",
    "transfer": "data movement",
    "compute": "compute",
    "other": "other",
}
PHASES = ("forward", "loss", "backward", "optimizer")


def _tree_rows() -> List[str]:
    """Every node of TREE (full keys) in report order, intermediates included."""
    rows: List[str] = []
    for cat, leaves in TREE.items():
        rows.append(cat)
        seen = set()
        for leaf in leaves:
            parts = leaf.split(".")
            for i in range(1, len(parts) + 1):
                key = f"{cat}." + ".".join(parts[:i])
                if key not in seen:
                    seen.add(key)
                    rows.append(key)
        rows.append(f"{cat}.{UNATTRIBUTED}")
    rows.append("other")
    return rows


TREE_ROWS = _tree_rows()


def _rollup(leaves: Dict[str, float]) -> Dict[str, float]:
    """{"ssd.read.gather.wait": x, "ssd": y} -> every level:
    "ssd.read.gather.wait", "ssd.read.gather", "ssd.read", "ssd",
    "ssd.(unattributed)"."""
    out: Dict[str, float] = defaultdict(float)
    for key, v in leaves.items():
        parts = key.split(".")
        if len(parts) == 1:
            out[f"{key}.{UNATTRIBUTED}"] += v
        for i in range(1, len(parts) + 1):
            out[".".join(parts[:i])] += v
    return dict(out)


class _Frame:
    __slots__ = ("key", "t0", "child_ns")

    def __init__(self, key: str, t0: int):
        self.key = key
        self.t0 = t0
        self.child_ns = 0


class _Timer:
    __slots__ = ("_bd", "_key", "_frame")

    def __init__(self, bd: "TimeBreakdown", key: str):
        self._bd = bd
        self._key = key
        self._frame: Optional[_Frame] = None

    def __enter__(self):
        if self._bd.active:
            self._frame = self._bd._push(self._key)
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
        self._counts: Dict[str, Dict[str, int]] = {}
        self._wall_ns: Dict[str, int] = {}
        self._snapshots: Dict[str, dict] = {}

    # ---- recording -----------------------------------------------------

    def timed(self, key: str) -> _Timer:
        """Context manager: time spent inside counts for `key`
        (a top-level category or one of its TREE leaves)."""
        assert key in KEYS, key
        return _Timer(self, key)

    def count(self, name: str, n: int = 1) -> None:
        """Add n to a per-phase counter (no-op outside an epoch's phases)."""
        if not self.active or self._phase is None:
            return
        with self._lock:
            self._counts[self._phase][name] += int(n)

    def snapshot(self, name: str, data: dict) -> None:
        """Store end-of-epoch state under `name` (e.g. ftl.stats())."""
        if self.active:
            self._snapshots[name] = dict(data)

    def begin_epoch(self) -> None:
        self._stack = []
        self._ns = {p: defaultdict(int) for p in PHASES}
        self._counts = {p: defaultdict(int) for p in PHASES}
        self._wall_ns = {p: 0 for p in PHASES}
        self._snapshots = {}
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
            wall = self._wall_ns[p] / 1e9
            detail = _rollup({k: v / 1e9 for k, v in self._ns[p].items()})
            top = {c: detail.get(c, 0.0) for c in CATEGORIES}
            top["other"] = max(0.0, wall - sum(top.values()))
            detail["other"] = top["other"]
            record["phases"][p] = {
                "wall_s": wall,
                **top,
                "detail": detail,
                "counts": dict(self._counts[p]),
            }
        # Setup outside the phases (buffer reset etc.) counts as other.
        phase_wall = sum(self._wall_ns.values()) / 1e9
        record["outside_phases_s"] = max(0.0, record["wall_s"] - phase_wall)
        record["snapshots"] = self._snapshots
        self.epochs.append(record)
        return record

    def _close_phase(self, now: int) -> None:
        if self._phase is not None:
            self._wall_ns[self._phase] += now - self._phase_t0

    def _push(self, key: str) -> _Frame:
        frame = _Frame(key, time.perf_counter_ns())
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
                self._ns[self._phase][frame.key] += max(0, total - frame.child_ns)
            if parent is not None:
                parent.child_ns += total

    # ---- totals --------------------------------------------------------

    @staticmethod
    def _totals(record: dict) -> Dict[str, float]:
        """Top-level categories summed over phases (+ outside time as other)."""
        out = {c: 0.0 for c in ALL_CATEGORIES}
        for p in PHASES:
            for c in out:
                out[c] += record["phases"][p].get(c, 0.0)
        out["other"] += record["outside_phases_s"]
        return out

    @staticmethod
    def _detail_totals(record: dict) -> Dict[str, float]:
        """Every tree key summed over phases (+ outside time as other)."""
        out: Dict[str, float] = defaultdict(float)
        for p in PHASES:
            for k, v in record["phases"][p].get("detail", {}).items():
                out[k] += v
        out["other"] += record["outside_phases_s"]
        return dict(out)

    @staticmethod
    def _count_totals(record: dict) -> Dict[str, int]:
        out: Dict[str, int] = defaultdict(int)
        for p in PHASES:
            for k, v in record["phases"][p].get("counts", {}).items():
                out[k] += v
        return dict(out)

    # ---- reporting -----------------------------------------------------

    def print_epoch(self, record: Optional[dict] = None, title: str = "") -> None:
        """Full tree: seconds, % of its top-level category, % of total time,
        per phase and in total. Then counters and snapshots."""
        if record is None:
            record = self.epochs[-1]
        columns = []  # (name, detail values, wall)
        for p in PHASES:
            ph = record["phases"][p]
            columns.append((p, ph.get("detail", {}), ph["wall_s"]))
        columns.append(("total", self._detail_totals(record), record["wall_s"]))

        key_w = 30
        cell_w = 20

        def cell(v: float, cat_v: float, wall: float, top: bool) -> str:
            if v <= 0.0:
                return f"{'-':>7}".ljust(cell_w)
            pc = "100%" if top else (f"{100 * v / cat_v:.0f}%" if cat_v > 0 else "-")
            pt = f"{100 * v / wall:.0f}%" if wall > 0 else "-"
            return f"{f'{v:.2f}s':>7} {pc:>4} {pt:>4}   "[:cell_w].ljust(cell_w)

        print(f"\n========== TIME BREAKDOWN {title} "
              f"(exposed time; s | % of category | % of total) ==========")
        print(f"{'key':<{key_w}}" + "".join(f"{name:<{cell_w}}" for name, _, _ in columns))
        for key in TREE_ROWS:
            parts = key.split(".")
            top = len(parts) == 1
            label = ("  " * (len(parts) - 1)) + parts[-1]
            cat = parts[0]
            cells = "".join(
                cell(d.get(key, 0.0), d.get(cat, 0.0), wall, top)
                for _, d, wall in columns
            )
            print(f"{label:<{key_w}}{cells}")
        print("-" * (key_w + cell_w * len(columns)))
        print(f"{'wall':<{key_w}}" + "".join(
            f"{f'{wall:.2f}s':>7} {'':>4} {'100%':>4}   "[:cell_w].ljust(cell_w)
            for _, _, wall in columns
        ))

        self._print_counts(record, key_w, cell_w)
        self._print_snapshots(record)

    def _print_counts(self, record: dict, key_w: int, cell_w: int) -> None:
        per_phase = [record["phases"][p].get("counts", {}) for p in PHASES]
        totals = self._count_totals(record)
        cols = [*per_phase, totals]
        if not totals:
            return
        print(f"\n{'counters':<{key_w}}" + "".join(f"{n:<{cell_w}}" for n in (*PHASES, "total")))

        def fmt(v) -> str:
            if v is None:
                return f"{'-':<{cell_w}}"
            if isinstance(v, float):
                return f"{v:<{cell_w}.1f}"
            return f"{v:<{cell_w},}"

        for name in sorted(totals):
            print(f"{name:<{key_w}}" + "".join(fmt(c.get(name) or None) for c in cols))

        def ratio(num: str, den: str, scale: float = 1.0):
            return [
                (scale * c[num] / c[den]) if c.get(den) else None
                for c in cols
            ]

        derived = []
        if "ftl.rows.in" in totals and "ftl.runs.out" in totals:
            derived.append(("  rows/run", ratio("ftl.rows.in", "ftl.runs.out")))
        if "ftl.lookup.hit" in totals or "ftl.lookup.miss" in totals:
            hit_pct = []
            for c in cols:
                n = c.get("ftl.lookup.hit", 0) + c.get("ftl.lookup.miss", 0)
                hit_pct.append(100.0 * c.get("ftl.lookup.hit", 0) / n if n else None)
            derived.append(("  lookup hit %", hit_pct))
        if "ftl.bytes.read" in totals:
            derived.append(("  GiB read (row runs)", [
                (c["ftl.bytes.read"] / 2**30) if c.get("ftl.bytes.read") else None
                for c in cols
            ]))
        for name, vals in derived:
            print(f"{name:<{key_w}}" + "".join(fmt(v) for v in vals))

    def _print_snapshots(self, record: dict) -> None:
        snaps = record.get("snapshots") or {}
        if not snaps:
            return
        prev = None
        if record in self.epochs:
            i = self.epochs.index(record)
            if i > 0:
                prev = self.epochs[i - 1].get("snapshots") or {}
        for name, data in snaps.items():
            keys = list(data)
            print(f"\n{name + ' state':<30}" + "".join(f"{k:>16}" for k in keys))
            print(f"{'':<30}" + "".join(f"{data[k]:>16,}" for k in keys))
            if prev and name in prev:
                p = prev[name]
                print(f"{'  change vs previous epoch':<30}" + "".join(
                    f"{data[k] - p.get(k, 0):>+16,}" for k in keys
                ))

    def print_summary(self, skip_first: bool = True) -> None:
        """Median over epochs (the first one skipped when there are more,
        since it includes warm-up). Percentages come from the median seconds."""
        epochs = self.epochs[1:] if skip_first and len(self.epochs) > 1 else self.epochs
        if not epochs:
            return
        med = statistics.median

        def med_dict(dicts: List[dict]) -> dict:
            keys = set().union(*dicts)
            return {k: med([d.get(k, 0) for d in dicts]) for k in keys}

        summary = {
            "wall_s": med([e["wall_s"] for e in epochs]),
            "outside_phases_s": med([e["outside_phases_s"] for e in epochs]),
            "phases": {},
            "snapshots": epochs[-1].get("snapshots", {}),
        }
        for p in PHASES:
            phs = [e["phases"][p] for e in epochs]
            summary["phases"][p] = {
                **{k: med([ph[k] for ph in phs]) for k in ("wall_s", *ALL_CATEGORIES)},
                "detail": med_dict([ph.get("detail", {}) for ph in phs]),
                "counts": med_dict([ph.get("counts", {}) for ph in phs]),
            }
        n = len(epochs)
        self.print_epoch(summary, title=f"median of {n} epoch{'s' if n > 1 else ''}")

    def save(self, path: str) -> None:
        """All epochs as JSON (seconds; percentages are left to the reader)."""
        with open(path, "w") as f:
            json.dump(
                {
                    "schema": 2,
                    "categories": {c: LABELS[c] for c in ALL_CATEGORIES},
                    "tree": {c: list(leaves) for c, leaves in TREE.items()},
                    "epochs": [
                        {
                            **e,
                            "totals": self._totals(e),
                            "detail_totals": self._detail_totals(e),
                            "count_totals": self._count_totals(e),
                        }
                        for e in self.epochs
                    ],
                },
                f,
                indent=2,
            )
