"""TimeBreakdown: nested dotted keys, rollups, counters, report and JSON."""

import json
import time

import pytest

from grinnder.stats.breakdown import TREE_ROWS, TimeBreakdown, _rollup


def _spin(seconds: float) -> None:
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


def _epoch(bd: TimeBreakdown) -> dict:
    bd.begin_epoch()
    bd.set_phase("forward")
    with bd.timed("ssd.read.gather.wait"):
        _spin(0.02)
        with bd.timed("ftl.translate.rows"):  # inner key wins
            _spin(0.01)
            bd.count("ftl.rows.in", 100)
            bd.count("ftl.runs.out", 10)
    with bd.timed("ssd"):  # bare top-level key
        _spin(0.005)
    bd.set_phase("backward")
    with bd.timed("compute.backward"):
        _spin(0.01)
    bd.count("ftl.lookup.hit", 3)
    bd.count("ftl.lookup.miss", 1)
    bd.snapshot("ftl", {"free_pages": 10, "invalid_pages": 2})
    return bd.end_epoch()


def test_rollup_levels():
    r = _rollup({"ssd.read.gather.wait": 1.0, "ssd.write.gradient.wait": 2.0, "ssd": 0.5})
    assert r["ssd.read.gather.wait"] == 1.0
    assert r["ssd.read.gather"] == 1.0
    assert r["ssd.read"] == 1.0
    assert r["ssd.write"] == 2.0
    assert r["ssd.(unattributed)"] == 0.5
    assert r["ssd"] == pytest.approx(3.5)


def test_unknown_key_rejected():
    with pytest.raises(AssertionError):
        TimeBreakdown().timed("ssd.nope")


def test_nested_attribution_and_totals():
    bd = TimeBreakdown()
    rec = _epoch(bd)
    fwd = rec["phases"]["forward"]
    d = fwd["detail"]
    # Inner FTL time is not counted again as SSD time.
    assert d["ftl.translate.rows"] == pytest.approx(0.01, abs=0.005)
    assert d["ssd.read.gather.wait"] == pytest.approx(0.02, abs=0.005)
    assert d["ssd"] == pytest.approx(d["ssd.read"] + d["ssd.(unattributed)"])
    assert fwd["ftl"] == d["ftl"]
    # Top-level categories add up to the phase wall time.
    top = sum(fwd[c] for c in ("ssd", "ftl", "transfer", "compute", "other"))
    assert top == pytest.approx(fwd["wall_s"])
    assert fwd["counts"] == {"ftl.rows.in": 100, "ftl.runs.out": 10}
    assert rec["phases"]["backward"]["counts"]["ftl.lookup.miss"] == 1
    assert rec["snapshots"]["ftl"]["free_pages"] == 10


def test_count_outside_epoch_is_ignored():
    bd = TimeBreakdown()
    bd.count("ftl.rows.in", 5)
    rec = _epoch(bd)
    assert TimeBreakdown._count_totals(rec)["ftl.rows.in"] == 100


def test_report_and_json(tmp_path, capsys):
    bd = TimeBreakdown()
    _epoch(bd)
    rec = _epoch(bd)
    rec["snapshots"]["ftl"]["free_pages"] = 7  # shows a change vs epoch 0
    bd.print_epoch(title="epoch 1")
    bd.print_summary()
    out = capsys.readouterr().out
    # Full tree, every row, every time.
    for key in TREE_ROWS:
        assert f" {key.split('.')[-1]}" in out or out.count(key.split(".")[-1])
    assert "% of category" in out
    assert "rows/run" in out and "10.0" in out
    assert "lookup hit %" in out and "75.0" in out
    assert "-3" in out  # free_pages change
    assert "median of 1 epoch" in out

    path = tmp_path / "bd.json"
    bd.save(str(path))
    data = json.loads(path.read_text())
    assert data["schema"] == 2
    e = data["epochs"][0]
    assert "ftl.translate.rows" in e["detail_totals"]
    assert e["count_totals"]["ftl.runs.out"] == 10
    assert set(e["totals"]) == {"ssd", "ftl", "transfer", "compute", "other"}


def test_timer_keys_in_code_exist():
    """Every literal key passed to timed() / stat.timed() in the package is
    a TREE key (dynamic keys are built from the same parts, checked below)."""
    import pathlib
    import re

    from grinnder.stats.breakdown import KEYS

    root = pathlib.Path(__file__).resolve().parents[1] / "grinnder"
    pat = re.compile(r"""\btimed\(\s*["']([^"']+)["']""")
    used = {
        m.group(1)
        for f in root.rglob("*.py")
        if f.name != "breakdown.py"
        for m in pat.finditer(f.read_text())
    }
    assert used, "no timer keys found"
    assert used <= KEYS, sorted(used - KEYS)

    dynamic = (
        [f"ssd.read.{k}.{w}" for k in ("partition", "gradient") for w in ("submit", "wait")]
        + [f"ssd.write.{k}.{w}" for k in ("activation", "gradient") for w in ("submit", "wait")]
        + [f"transfer.d2h.{k}" for k in ("activation", "gradient")]
        + [f"transfer.h2d.{k}" for k in ("features", "gradient")]
        + ["compute.recompute", "compute.forward", "compute.grad_sum"]
    )
    assert set(dynamic) <= KEYS, sorted(set(dynamic) - KEYS)


def test_timed_by_picks_key_per_call():
    from grinnder.stats import stat, timed_by

    class Buf:
        def __init__(self, kind):
            self.key = f"transfer.d2h.{kind}"

        @timed_by(lambda self, *a, **k: self.key)
        def sync(self):
            _spin(0.005)

    bd = stat.breakdown
    bd.begin_epoch()
    bd.set_phase("backward")
    Buf("activation").sync()
    Buf("gradient").sync()
    rec = bd.end_epoch()
    bd.epochs.clear()
    d = rec["phases"]["backward"]["detail"]
    assert d["transfer.d2h.activation"] > 0
    assert d["transfer.d2h.gradient"] > 0
    assert d["transfer.d2h"] == pytest.approx(
        d["transfer.d2h.activation"] + d["transfer.d2h.gradient"]
    )
