"""Cross-partition dependency analysis for full-graph GNN training.

Answers two questions for a given partitioning:

  1. How concentrated are each partition's dependencies?
     For every partition: how many source partitions are needed to cover
     80% / 90% / 99% / 100% of its boundary rows (100% = exact training).

  2. How much does partition-granularity caching over-read?
     Rows actually needed (own + boundary rows) vs rows loaded when every
     source partition with at least one dependency must be loaded whole.

Dependencies are recomputed from the raw edges and the partition assignment,
not from cached `boundaries` (those may have been pruned with --sparse 1).

Usage:
  # Fresh METIS partitioning for several partition counts
  python examples/analyze_partition_deps.py --igb_root data/igb_datasets \
      --metis --num_parts 16 32 64 --out_dir results/deps

  # Reuse the partition assignment (perm/ptr) of an existing cache file
  python examples/analyze_partition_deps.py --igb_root data/igb_datasets \
      --cache data/grinnder_partitions/igb_medium_grinnder_64p.pt --out_dir results/deps
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from grinnder.data.datasets import load_igb

COVERAGE_LEVELS = (0.80, 0.90, 0.99, 1.00)


# --------------------------------------------------------------------------
# Partition assignment
# --------------------------------------------------------------------------

def metis_assignment(edge_index, num_nodes, num_parts):
    """Run METIS the same way build_partitioned_graph_metis does."""
    from torch_geometric.data import Data
    from torch_geometric.loader import ClusterData

    data = Data(edge_index=edge_index, num_nodes=num_nodes)
    cd = ClusterData(data, num_parts=num_parts, recursive=False, log=False)
    return cd.partition.node_perm.cpu(), cd.partition.partptr.cpu()


def cache_assignment(cache_path):
    try:
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    except TypeError:
        cache = torch.load(cache_path, map_location="cpu")
    print(f"  cache partitioner field = {cache.get('partitioner')!r} "
          f"(note: METIS runs also store 'grinnder' here)")
    return cache["perm"].cpu(), cache["ptr"].cpu()


# --------------------------------------------------------------------------
# Dependency matrix
# --------------------------------------------------------------------------

def dependency_matrix(edge_index, perm, ptr):
    """dep[p, q] = number of distinct rows of partition q that partition p needs.

    Matches _build_partition_subgraph: partition(row) needs the col endpoint.
    """
    num_nodes = perm.numel()
    num_parts = ptr.numel() - 1

    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(num_nodes, dtype=perm.dtype)

    row = inv_perm[edge_index[0]]
    col = inv_perm[edge_index[1]]

    # Node id (after reordering) -> partition id
    part_row = torch.searchsorted(ptr, row, right=True) - 1
    part_col = torch.searchsorted(ptr, col, right=True) - 1
    del row

    cross = part_row != part_col
    part_row = part_row[cross]
    part_col = part_col[cross]
    col = col[cross]

    # Distinct (needing partition, needed node) pairs
    key = torch.unique(part_row * num_nodes + col)
    p = key // num_nodes
    q = torch.searchsorted(ptr, key % num_nodes, right=True) - 1

    dep = torch.zeros(num_parts * num_parts, dtype=torch.long)
    dep.scatter_add_(0, p * num_parts + q, torch.ones_like(p))
    return dep.view(num_parts, num_parts).numpy()


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def sources_for_coverage(counts, level):
    """Smallest number of source partitions covering `level` of the boundary rows."""
    total = counts.sum()
    if total == 0:
        return 0
    if level >= 1.0:
        return int((counts > 0).sum())
    c = np.sort(counts)[::-1].cumsum()
    return int(np.searchsorted(c, level * total - 1e-9) + 1)


def analyze(dep, part_sizes, prune_pct, row_bytes):
    num_parts = dep.shape[0]
    rows = []
    for p in range(num_parts):
        counts = dep[p].copy()
        counts[p] = 0
        boundary = int(counts.sum())
        own = int(part_sizes[p])

        srcs_any = counts > 0
        srcs_pruned = counts >= prune_pct / 100.0 * boundary if boundary else srcs_any

        top10 = np.sort(counts)[::-1][:10].sum() / boundary if boundary else 1.0

        needed = own + boundary
        loaded_exact = own + int(part_sizes[srcs_any].sum())
        loaded_pruned = own + int(part_sizes[srcs_pruned].sum())

        r = {
            "partition": p,
            "own_rows": own,
            "boundary_rows": boundary,
            "boundary_over_own": boundary / own if own else 0.0,
            "sources_any": int(srcs_any.sum()),
            f"sources_ge_{prune_pct:g}pct": int(srcs_pruned.sum()),
            "top10_share": float(top10),
        }
        for lvl in COVERAGE_LEVELS:
            r[f"sources_{int(lvl * 100)}pct"] = sources_for_coverage(counts, lvl)
        r.update({
            "rows_needed": needed,
            "rows_loaded_partition_cache": loaded_exact,
            "rows_loaded_partition_cache_pruned": loaded_pruned,
            "overread_exact": loaded_exact / needed,
            "overread_pruned": loaded_pruned / needed,
            "gb_needed": needed * row_bytes / 1024**3,
            "gb_loaded_partition_cache": loaded_exact * row_bytes / 1024**3,
        })
        rows.append(r)
    return rows


def summarize(rows, num_parts, prune_pct):
    def stats(key):
        v = np.array([r[key] for r in rows], dtype=float)
        return {"min": float(v.min()), "median": float(np.median(v)),
                "mean": float(v.mean()), "max": float(v.max())}

    keys = ["boundary_over_own", "sources_any", f"sources_ge_{prune_pct:g}pct",
            "top10_share", "overread_exact", "overread_pruned"]
    keys += [f"sources_{int(l * 100)}pct" for l in COVERAGE_LEVELS]
    s = {k: stats(k) for k in keys}
    s["num_parts"] = num_parts
    s["partitions_where_top10_covers_80pct"] = sum(r["top10_share"] >= 0.80 for r in rows)
    s["total_rows_needed"] = sum(r["rows_needed"] for r in rows)
    s["total_rows_loaded_partition_cache"] = sum(r["rows_loaded_partition_cache"] for r in rows)
    return s


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def print_report(rows, summary, prune_pct):
    n = summary["num_parts"]
    print(f"\n{'=' * 100}\n  {n} PARTITIONS\n{'=' * 100}")
    hdr = (f"{'P':>3} {'own':>8} {'bndry':>8} {'b/own':>5} | {'any':>3} {'>=' + format(prune_pct, 'g') + '%':>4} "
           f"{'80%':>3} {'90%':>3} {'99%':>3} | {'top10':>5} | {'needed':>8} {'loaded':>9} {'x':>5}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['partition']:>3} {r['own_rows']:>8,} {r['boundary_rows']:>8,} {r['boundary_over_own']:>5.2f} | "
              f"{r['sources_any']:>3} {r[f'sources_ge_{prune_pct:g}pct']:>4} "
              f"{r['sources_80pct']:>3} {r['sources_90pct']:>3} {r['sources_99pct']:>3} | "
              f"{r['top10_share']:>5.0%} | {r['rows_needed']:>8,} {r['rows_loaded_partition_cache']:>9,} "
              f"{r['overread_exact']:>5.1f}")

    print("\nSummary (median [min-max]):")
    for k in ["boundary_over_own", "sources_any", f"sources_ge_{prune_pct:g}pct", "sources_80pct",
              "sources_90pct", "sources_99pct", "top10_share", "overread_exact", "overread_pruned"]:
        v = summary[k]
        print(f"  {k:<28} {v['median']:>7.2f}  [{v['min']:.2f} - {v['max']:.2f}]")
    print(f"  partitions where top-10 sources cover >=80%: "
          f"{summary['partitions_where_top10_covers_80pct']}/{n}")
    tot_n, tot_l = summary["total_rows_needed"], summary["total_rows_loaded_partition_cache"]
    print(f"  per-layer rows needed (sum over partitions):        {tot_n:,}")
    print(f"  per-layer rows loaded with partition-level caching: {tot_l:,} ({tot_l / tot_n:.1f}x)")


def write_outputs(rows, summary, out_dir, tag):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"deps_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / f"deps_{tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def plot(all_results, out_dir, prune_pct):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots (CSVs written).")
        return

    # Figure 1: CDF over partitions of #source partitions needed per coverage level
    fig, axes = plt.subplots(1, len(all_results), figsize=(4.2 * len(all_results), 3.4),
                             squeeze=False)
    for ax, (n, rows) in zip(axes[0], all_results.items()):
        for key, label in [("sources_80pct", "80% of rows"), ("sources_90pct", "90%"),
                           ("sources_99pct", "99%"), ("sources_any", "100% (exact)")]:
            v = np.sort([r[key] for r in rows])
            ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", label=label)
        ax.set_title(f"{n} partitions")
        ax.set_xlabel("source partitions needed")
        ax.set_xlim(0, n - 1)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("fraction of partitions")
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "deps_coverage_cdf.png", dpi=200)
    plt.close(fig)

    # Figure 2: rows needed vs rows loaded with partition-granularity caching
    fig, axes = plt.subplots(len(all_results), 1, figsize=(10, 2.8 * len(all_results)),
                             squeeze=False)
    for ax, (n, rows) in zip(axes[:, 0], all_results.items()):
        x = np.arange(len(rows))
        own = np.array([r["own_rows"] for r in rows]) / 1e6
        bnd = np.array([r["boundary_rows"] for r in rows]) / 1e6
        loaded = np.array([r["rows_loaded_partition_cache"] for r in rows]) / 1e6
        ax.bar(x, loaded, color="0.85", label="loaded (partition-level cache)")
        ax.bar(x, own, color="C0", label="own rows")
        ax.bar(x, bnd, bottom=own, color="C1", label="boundary rows needed")
        ax.set_title(f"{n} partitions")
        ax.set_ylabel("rows per layer (M)")
        ax.set_xlim(-0.6, len(rows) - 0.4)
    axes[-1, 0].set_xlabel("partition")
    axes[0, 0].legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out_dir / "deps_needed_vs_loaded.png", dpi=200)
    plt.close(fig)
    print(f"Plots written to {out_dir}")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--igb_root", default="data/igb_datasets")
    ap.add_argument("--igb_size", default="medium", choices=["tiny", "small", "medium", "large"])
    ap.add_argument("--num_classes", type=int, default=19)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--metis", action="store_true", help="partition with METIS")
    src.add_argument("--cache", type=str, help="use perm/ptr from a partition cache file")
    ap.add_argument("--num_parts", type=int, nargs="+", default=[64],
                    help="partition counts to analyze (with --metis)")
    ap.add_argument("--prune_pct", type=float, default=1.0,
                    help="threshold matching --sparse 1 pruning (percent of boundary rows)")
    ap.add_argument("--row_bytes", type=int, default=None,
                    help="bytes per row for GB columns (default: feature dim * 4)")
    ap.add_argument("--out_dir", default="results/partition_deps")
    args = ap.parse_args()

    print(f"Loading IGB-{args.igb_size} edges...")
    data = load_igb(args.igb_root, size=args.igb_size, num_classes=args.num_classes,
                    mmap_features=True, download=False)
    edge_index = data.edge_index.cpu()
    num_nodes = data.num_nodes
    row_bytes = args.row_bytes or int(data.x.size(1)) * 4
    print(f"  nodes={num_nodes:,} edges={edge_index.size(1):,} row_bytes={row_bytes}")

    out_dir = Path(args.out_dir)
    all_results = {}

    runs = args.num_parts if args.metis else [None]
    for n in runs:
        if args.metis:
            print(f"\nMETIS -> {n} partitions...")
            perm, ptr = metis_assignment(edge_index, num_nodes, n)
            tag = f"{args.igb_size}_metis_{n}p"
        else:
            print(f"\nLoading partition assignment from {args.cache}")
            perm, ptr = cache_assignment(args.cache)
            n = ptr.numel() - 1
            tag = f"{args.igb_size}_{Path(args.cache).stem}"

        part_sizes = (ptr[1:] - ptr[:-1]).numpy()
        dep = dependency_matrix(edge_index, perm, ptr)
        rows = analyze(dep, part_sizes, args.prune_pct, row_bytes)
        summary = summarize(rows, n, args.prune_pct)

        print_report(rows, summary, args.prune_pct)
        write_outputs(rows, summary, out_dir, tag)
        np.save(out_dir / f"deps_{tag}_matrix.npy", dep)
        all_results[n] = rows

    plot(all_results, out_dir, args.prune_pct)
    print(f"\nCSV / JSON / matrix written to {out_dir}")


if __name__ == "__main__":
    main()
