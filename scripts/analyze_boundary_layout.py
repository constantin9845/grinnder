#!/usr/bin/env python3
"""Offline check for option 3: group boundary rows inside each partition by
which partitions need them, and count the SSD pages and reads that boundary
gathers would need. Reads only the partition cache (no GPU, no SSD).

Layouts compared, for each row size:
  as-is     current row order, rows requested in discovery order (today's reads)
  sorted    current row order, boundary lists sorted (option 2)
  opt3-cnt  option 3, rows ordered by (number of consumers, consumer set), lists sorted
  opt3-min  option 3, rows ordered by (lowest consumer, number of consumers, consumer set), lists sorted
  ideal     every (target, source) pair fully contiguous: the lower bound for any layout

Columns:
  pages       flash pages that hold at least one needed row (NAND page reads)
  rows/page   needed rows per page read (higher is better)
  amp         bytes read from flash / bytes needed
  row reads   I/Os when rows go straight into the GPU tensor: rows next to each
              other on the SSD and in the request merge (as _merge_runs), max 128 KiB
  page reads  I/Os when whole pages are read (as --read_coalesce), touching pages merged

Assumes a partition's rows sit on consecutive SSD pages (the FTL allocates a
tensor's pages in order), so "next row" on the SSD = next local row index.

Example:
  python scripts/analyze_boundary_layout.py \
      --cache data/grinnder_partitions/igb_medium_grinnder_64p.pt --pid 0
"""

import argparse

import numpy as np
import torch

MAX_IO = 128 * 1024          # GPU_WRITE_MAX_IO in async_transfer.cu


def load_cache(path):
    for kwargs in (dict(weights_only=False, mmap=True), dict(weights_only=False), {}):
        try:
            return torch.load(path, map_location="cpu", **kwargs)
        except TypeError:
            continue
    raise RuntimeError(f"could not load {path}")


def build_needs(boundaries, num_parts):
    """needs[src] = [(dst, local row indices in src that dst needs, request order)]."""
    needs = [[] for _ in range(num_parts)]
    for dst in range(num_parts):
        row = boundaries[dst]
        if row is None:
            continue
        for src in range(num_parts):
            b = row[src]
            if src == dst or b is None or b.numel() == 0:
                continue
            needs[src].append((dst, b.cpu().numpy().astype(np.int64, copy=False)))
    return needs


def option3_positions(n, src_needs, num_parts, mode):
    """New local position of every row of one source partition under option 3."""
    used = np.zeros((n, num_parts), dtype=bool)
    for dst, idx in src_needs:
        used[idx, dst] = True
    ncons = used.sum(axis=1)
    packed = np.packbits(used, axis=1, bitorder="little")
    # np.lexsort: the LAST key is the primary one; ties keep the old order.
    keys = [packed[:, c] for c in range(packed.shape[1] - 1, -1, -1)] + [ncons]
    if mode == "min":
        lowest = np.where(ncons > 0, used.argmax(axis=1), -1)
        keys.append(lowest)
    order = np.lexsort(tuple(keys))      # interior rows (ncons 0) come first
    pos = np.empty(n, dtype=np.int64)
    pos[order] = np.arange(n, dtype=np.int64)
    return pos, ncons


def run_lengths(sorted_or_ordered):
    """Lengths of runs of +1 steps."""
    n = sorted_or_ordered.size
    if n == 0:
        return np.empty(0, dtype=np.int64)
    breaks = np.flatnonzero(np.diff(sorted_or_ordered) != 1) + 1
    return np.diff(np.concatenate(([0], breaks, [n])))


def ios(lengths, unit_bytes):
    return int(((lengths * unit_bytes + MAX_IO - 1) // MAX_IO).sum())


def pair_stats(rows, rpp, row_bytes, page_bytes):
    pages = np.unique(rows // rpp)
    return np.array([
        rows.size,
        pages.size,
        ios(run_lengths(rows), row_bytes),
        ios(run_lengths(pages), page_bytes),
    ], dtype=np.int64)


def ideal_stats(n, rpp, row_bytes, page_bytes):
    pages = -(-n // rpp)
    return np.array([
        n,
        pages,
        -(-(n * row_bytes) // MAX_IO),
        -(-(pages * page_bytes) // MAX_IO),
    ], dtype=np.int64)


def print_table(title, stats, row_bytes, page_bytes):
    print(f"\n{title}")
    print(f"  {'layout':<9} {'rows':>12} {'pages':>12} {'rows/page':>9} {'amp':>6} "
          f"{'row reads':>12} {'page reads':>12}")
    for name, (rows, pages, row_ios, page_ios) in stats.items():
        rpp_used = rows / pages if pages else 0.0
        amp = pages * page_bytes / (rows * row_bytes) if rows else 0.0
        print(f"  {name:<9} {rows:>12,} {pages:>12,} {rpp_used:>9.2f} {amp:>6.2f} "
              f"{row_ios:>12,} {page_ios:>12,}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="partition cache .pt file")
    ap.add_argument("--row-bytes", type=int, nargs="+", default=[4096, 1024],
                    help="row sizes to test (default: 4096 = 1024 fp32 features, "
                         "1024 = hidden 256 fp32)")
    ap.add_argument("--page-bytes", type=int, default=16384, help="flash page size (default 16384)")
    ap.add_argument("--pid", type=int, default=None,
                    help="also print one target partition (compare with the [gather_*] log lines)")
    args = ap.parse_args()

    cache = load_cache(args.cache)
    num_parts = int(cache["num_parts"])
    sizes = [int(s) for s in cache["partition_sizes"]]
    print(f"{args.cache}: {num_parts} partitions, {sum(sizes):,} nodes")

    needs = build_needs(cache["boundaries"], num_parts)

    # Option 3 layouts, and how many partitions need each boundary row.
    pos = {"opt3-cnt": [], "opt3-min": []}
    hist = np.zeros(num_parts + 1, dtype=np.int64)
    for src in range(num_parts):
        p_cnt, ncons = option3_positions(sizes[src], needs[src], num_parts, "cnt")
        p_min, _ = option3_positions(sizes[src], needs[src], num_parts, "min")
        pos["opt3-cnt"].append(p_cnt)
        pos["opt3-min"].append(p_min)
        hist += np.bincount(ncons, minlength=num_parts + 1)

    boundary_nodes = hist[1:].sum()
    print(f"boundary nodes: {boundary_nodes:,} of {sum(sizes):,} "
          f"({100 * boundary_nodes / max(1, sum(sizes)):.1f}%)")
    print("needed by k partitions:  " + "  ".join(
        f"k={k}: {100 * hist[k] / max(1, boundary_nodes):.1f}%"
        for k in range(1, min(num_parts, 5) + 1))
        + f"  k>5: {100 * hist[6:].sum() / max(1, boundary_nodes):.1f}%")

    for row_bytes in args.row_bytes:
        if args.page_bytes % row_bytes:
            print(f"\nskip row size {row_bytes}: does not divide the page size")
            continue
        rpp = args.page_bytes // row_bytes
        names = ["as-is", "sorted", "opt3-cnt", "opt3-min", "ideal"]
        total = {k: np.zeros(4, dtype=np.int64) for k in names}
        one = {k: np.zeros(4, dtype=np.int64) for k in names}

        for src in range(num_parts):
            for dst, idx in needs[src]:
                s = {
                    "as-is": pair_stats(idx, rpp, row_bytes, args.page_bytes),
                    "sorted": pair_stats(np.sort(idx), rpp, row_bytes, args.page_bytes),
                    "opt3-cnt": pair_stats(np.sort(pos["opt3-cnt"][src][idx]), rpp,
                                           row_bytes, args.page_bytes),
                    "opt3-min": pair_stats(np.sort(pos["opt3-min"][src][idx]), rpp,
                                           row_bytes, args.page_bytes),
                    "ideal": ideal_stats(idx.size, rpp, row_bytes, args.page_bytes),
                }
                for k in names:
                    total[k] += s[k]
                    if dst == args.pid:
                        one[k] += s[k]

        print_table(f"row {row_bytes} B ({rpp} rows/page), all partitions:",
                    total, row_bytes, args.page_bytes)
        if args.pid is not None:
            print_table(f"row {row_bytes} B, target partition {args.pid} only:",
                        one, row_bytes, args.page_bytes)


if __name__ == "__main__":
    main()
