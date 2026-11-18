"""Trace (--trace): Python side, with a stand-in for the C++ event buffer."""

import json
import threading

from grinnder.stats import stat, trace as global_trace
from grinnder.stats.trace import Trace


class FakeOps:
    """Same API as the trace functions in grinnder._C."""

    def __init__(self):
        self.on = False
        self.buf = []
        self.seq = 0
        self.lock = threading.Lock()

    def trace_enable(self, on):
        self.on = on

    def trace_event(self, name, args):
        if not self.on:
            return
        with self.lock:
            self.buf.append((self.seq, 1000 + self.seq, threading.get_native_id(), name, args))
            self.seq += 1

    def trace_pending(self):
        return len(self.buf)

    def trace_drain(self):
        with self.lock:
            out, self.buf = self.buf, []
        return out


def _started(tmp_path):
    t = Trace()
    t._ops = FakeOps()
    t._pattern = str(tmp_path / "run_trace_epoch{}.jsonl")
    t.enabled = True
    return t


def _read(path):
    return [json.loads(line) for line in open(path)]


def test_off_by_default_records_nothing():
    t = Trace()
    t.event("partition", pid=1)   # no ops needed when off
    t.begin_epoch()
    assert t.end_epoch() is None


def test_epoch_file_in_order(tmp_path):
    t = _started(tmp_path)
    t.begin_epoch()
    t.event("partition", phase="forward", layer=0, pid=10)
    t._ops.trace_event("gds.submit", '{"job":0,"op":"read"}')   # as C++ would
    t.event("job", job=0, what="gather", pid=10)
    path = t.end_epoch()
    assert path.endswith("run_trace_epoch1.jsonl")
    evs = _read(path)
    assert [e["ev"] for e in evs] == ["epoch.begin", "partition", "gds.submit", "job", "epoch.end"]
    assert [e["seq"] for e in evs] == sorted(e["seq"] for e in evs)
    assert evs[1]["args"] == {"phase": "forward", "layer": 0, "pid": 10}
    assert evs[2]["args"]["op"] == "read"


def test_between_epochs_dropped_and_numbered(tmp_path):
    t = _started(tmp_path)
    t.begin_epoch()
    t.end_epoch()
    t._ops.trace_event("gds.io_done", "{}")   # recording is off: dropped
    t.event("partition", pid=1)
    t.begin_epoch()
    path = t.end_epoch()
    assert path.endswith("epoch2.jsonl")
    assert [e["ev"] for e in _read(path)] == ["epoch.begin", "epoch.end"]


def test_breakdown_timers_and_phase_logged(tmp_path, monkeypatch):
    fake = _started(tmp_path)
    for k in ("enabled", "_ops", "_pattern"):
        monkeypatch.setattr(global_trace, k, getattr(fake, k))
    monkeypatch.setattr(global_trace, "_epoch", 0)
    monkeypatch.setattr(global_trace, "paths", [])
    bd = stat.breakdown
    bd.begin_epoch()
    global_trace.begin_epoch()
    bd.set_phase("forward")
    with bd.timed("ssd.read.gather.wait"):
        with bd.timed("ftl.translate.rows"):
            pass
    bd.end_epoch()
    path = global_trace.end_epoch()
    bd.epochs.clear()
    evs = [(e["ev"], e["args"].get("key") or e["args"].get("phase")) for e in _read(path)]
    assert evs[1:-1] == [
        ("phase", "forward"),
        ("begin", "ssd.read.gather.wait"),
        ("begin", "ftl.translate.rows"),
        ("end", "ftl.translate.rows"),
        ("end", "ssd.read.gather.wait"),
    ]
