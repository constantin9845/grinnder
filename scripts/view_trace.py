#!/usr/bin/env python3
"""Print a --trace file (…_trace_epoch<N>.jsonl) as a readable, chronological list.

    python scripts/view_trace.py time_breakdown_trace_epoch1.jsonl | less -S

One line per event, in `seq` order:
    seq  [thread]  event  key=value ...

Threads: "main" is the thread that started the epoch, "job17" the cuFile
worker of job 17, other threads (e.g. autograd) are "t1", "t2", ...
Breakdown timers (begin/end) indent the events inside them.

Options:
    --no-io          hide gds.io_done lines (one per SSD I/O)
    --no-timers      hide begin/end timer lines
    --job 17         only events of job 17 (Python "job" line included)
    --pid 10         only from "partition pid=10" to the next partition event
    --from / --to    seq range
    --time           add the time since the epoch began (ms)
"""

from __future__ import annotations

import argparse
import json
import sys


def fmt_args(args: dict, skip=()) -> str:
    out = []
    for k, v in args.items():
        if k in skip:
            continue
        if isinstance(v, dict):
            v = ",".join(f"{a}:{b}" for a, b in v.items())
        elif isinstance(v, list):
            v = ",".join(map(str, v))
        out.append(f"{k}={v}")
    return " ".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("path")
    ap.add_argument("--no-io", action="store_true")
    ap.add_argument("--no-timers", action="store_true")
    ap.add_argument("--job", type=int)
    ap.add_argument("--pid", type=int)
    ap.add_argument("--from", dest="first", type=int, default=0)
    ap.add_argument("--to", dest="last", type=int)
    ap.add_argument("--time", action="store_true")
    a = ap.parse_args()

    with open(a.path) as f:
        evs = [json.loads(line) for line in f if line.strip()]
    evs.sort(key=lambda e: e["seq"])
    if not evs:
        print("empty trace")
        return 0

    # Thread names
    names = {evs[0]["tid"]: "main"}
    for e in evs:
        if e["ev"] == "gds.job_start":
            names.setdefault(e["tid"], f"job{e['args']['job']}")
    n_other = 0
    for e in evs:
        if e["tid"] not in names:
            n_other += 1
            names[e["tid"]] = f"t{n_other}"
    width = max(len(n) for n in names.values())

    t0 = evs[0]["t_ns"]
    depth = {}          # tid -> timer nesting
    in_pid = a.pid is None
    shown = 0
    out = sys.stdout
    for e in evs:
        ev, args, tid = e["ev"], e["args"], e["tid"]
        d = depth.get(tid, 0)
        if ev == "end":
            d = depth[tid] = max(0, d - 1)

        if ev == "partition" and a.pid is not None:
            in_pid = args.get("pid") == a.pid
        keep = (
            in_pid
            and a.first <= e["seq"] <= (a.last if a.last is not None else e["seq"])
            and not (a.no_io and ev == "gds.io_done")
            and not (a.no_timers and ev in ("begin", "end"))
            and (a.job is None or args.get("job") == a.job)
        )
        if keep:
            t = f"{(e['t_ns'] - t0) / 1e6:10.3f}ms " if a.time else ""
            if ev in ("begin", "end"):
                text = f"{ev} {args.get('key')}"
            else:
                text = f"{ev}  {fmt_args(args)}"
            out.write(f"{e['seq']:>8} {t}[{names[tid]:<{width}}] {'  ' * d}{text}\n")
            shown += 1

        if ev == "begin":
            depth[tid] = d + 1
    print(f"-- {shown} of {len(evs)} events", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:   # e.g. | head
        sys.exit(0)
