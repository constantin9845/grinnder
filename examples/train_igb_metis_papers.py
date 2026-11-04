"""Train GCN or GAT on ogbn-papers100M (111M nodes, 1.6B edges) with GriNNder.

Same pipeline as train_igb_metis.py (METIS partitioning, same flags, same
output), only the dataset differs.

Data preparation (one-time, cached under <ogb_root>/ogbn_papers100M/grinnder_npy/):
  1. The OGB dataset is downloaded (~57GB of features) and converted into
     plain .npy files (~100GB RAM).
  2. The edges are made undirected: both directions, duplicates and
     self-loops removed (~3.2B edges, ~60GB RAM). GCN normalization and METIS
     both assume an undirected graph; --no-undirected keeps the raw directed
     citations instead.
Features and edges are mmap-ed afterwards, not held in RAM.

Partitioning needs far more memory than training (~250GB peak for the
undirected graph), so it can run separately with --preprocess_only on a
machine with enough RAM. It saves the partition cache; the training run (with
the same --num_parts / --undirected) then loads it instead of partitioning.
It also prints the SSD space a training epoch needs.

Defaults (--num_parts 128, --hidden 128) keep partitions GPU-sized and the
per-epoch SSD footprint under ~400GB (base). Hidden 256 needs ~590GB.

Usage:
  # big-memory machine: prepare data + partition, no GPU training
  python examples/train_igb_metis_papers.py --ogb_root data/ogb_datasets --preprocess_only
  # training VM: copy <ogb_root>/ogbn_papers100M/grinnder_npy and the partition cache, then
  python examples/train_igb_metis_papers.py --ogb_root data/ogb_datasets --io_mode base --cache-size 16GB
"""

import argparse
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from grinnder import GAT, GCN, GriNNderConfig, Trainer, build_partitioned_graph, build_partitioned_graph_metis
from grinnder.data.datasets import NumpyFeatureStore
from grinnder.utils import fix_seed, get_default_partitioner_threads, report_memory
from grinnder.stats import stat


PAPERS_NUM_CLASSES = 172
_NPY_FILES = ("node_feat.npy", "edge_index.npy", "labels.npy",
              "train_idx.npy", "valid_idx.npy", "test_idx.npy")


def _convert_papers100m_to_npy(root: Path, out_dir: Path, download: bool, confirm_download: bool) -> None:
    """One-time conversion of the OGB dataset into mmap-able .npy files."""
    from ogb.nodeproppred import NodePropPredDataset
    import ogb.nodeproppred.dataset as ogb_dataset_mod

    if not download and not (root / "ogbn_papers100M").exists():
        raise FileNotFoundError(f"ogbn-papers100M not found under {root} (and --no-download was given)")
    if confirm_download:
        # OGB asks y/N on stdin before large downloads; answer yes for it.
        ogb_dataset_mod.decide_download = lambda url: True

    dataset = NodePropPredDataset(name="ogbn-papers100M", root=str(root))
    split_idx = dataset.get_idx_split()
    graph, labels = dataset[0]

    # Only ~1.5M nodes are labeled; the rest are NaN. Those nodes are in no
    # split, so give them class 0 to keep labels a valid int64 tensor
    # (num_classes is derived from labels.max()).
    labels = np.nan_to_num(labels.reshape(-1), nan=0.0).astype(np.int64)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Writing .npy files to {out_dir} (one-time)...")
    np.save(out_dir / "labels.npy", labels)
    np.save(out_dir / "train_idx.npy", np.asarray(split_idx["train"], dtype=np.int64))
    np.save(out_dir / "valid_idx.npy", np.asarray(split_idx["valid"], dtype=np.int64))
    np.save(out_dir / "test_idx.npy", np.asarray(split_idx["test"], dtype=np.int64))
    np.save(out_dir / "edge_index.npy", np.ascontiguousarray(graph["edge_index"], dtype=np.int64))
    # Written last: its presence marks a finished conversion.
    np.save(out_dir / "node_feat.npy", graph["node_feat"])


def _make_undirected(npy_dir: Path, num_nodes: int) -> None:
    """edge_index.npy (directed) -> edge_index_undirected.npy.

    Every edge in both directions, duplicates and self-loops removed (the
    partitioner adds exactly one self-loop per node). Works on one int64 key
    per edge (src * N + dst), sorted in place: peak RAM ~ 2 x 8 bytes per
    undirected edge (~55GB for papers100M).
    """
    src_path = npy_dir / "edge_index.npy"
    out_path = npy_dir / "edge_index_undirected.npy"
    print(f"  Making edges undirected (one-time) -> {out_path}")
    ei = np.load(src_path, mmap_mode="r")
    e = ei.shape[1]
    n = np.int64(num_nodes)  # N^2 < 2^63 for N = 111M
    step = 100_000_000

    keys = np.empty(2 * e, dtype=np.int64)
    for s in range(0, e, step):
        t = min(s + step, e)
        src = np.asarray(ei[0, s:t], dtype=np.int64)
        dst = np.asarray(ei[1, s:t], dtype=np.int64)
        keys[s:t] = src * n + dst
        keys[e + s:e + t] = dst * n + src
    del ei
    keys.sort()

    # Keep the first of each run of equal keys, and drop self-loops
    # (v * N + v = v * (N + 1): exactly the multiples of N + 1 below N^2).
    keep = np.empty(keys.shape[0], dtype=bool)
    keep[0] = True
    np.not_equal(keys[1:], keys[:-1], out=keep[1:])
    for s in range(0, keys.shape[0], step):
        keep[s:s + step] &= (keys[s:s + step] % (n + 1)) != 0
    keys = keys[keep]
    del keep

    m = keys.shape[0]
    tmp_path = out_path.with_suffix(".tmp.npy")
    out = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.int64, shape=(2, m))
    for s in range(0, m, step):
        k = keys[s:s + step]
        out[0, s:s + step] = k // n
        out[1, s:s + step] = k % n
    out.flush()
    del out
    os.replace(tmp_path, out_path)
    print(f"  {e:,} directed edges -> {m:,} undirected (both directions stored)")


def load_papers100m(
    root: str,
    mmap_features: bool = True,
    download: bool = True,
    confirm_download: bool = False,
    undirected: bool = True,
) -> Data:
    """Load ogbn-papers100M in the same form load_igb() returns."""
    root_path = Path(root)
    npy_dir = root_path / "ogbn_papers100M" / "grinnder_npy"
    if not all((npy_dir / f).is_file() for f in _NPY_FILES):
        _convert_papers100m_to_npy(root_path, npy_dir, download, confirm_download)

    feat = np.load(npy_dir / "node_feat.npy", mmap_mode="r" if mmap_features else None)
    num_nodes = feat.shape[0]

    edge_file = npy_dir / ("edge_index_undirected.npy" if undirected else "edge_index.npy")
    if undirected and not edge_file.is_file():
        _make_undirected(npy_dir, num_nodes)

    x = NumpyFeatureStore(feat) if mmap_features else torch.from_numpy(np.asarray(feat)).float()
    y = torch.from_numpy(np.load(npy_dir / "labels.npy")).long()
    # mmap (copy-on-write) instead of loading 26-52GB: partitioning makes its
    # own copies, and loading a cached partition only needs the edge count.
    edge_index = torch.from_numpy(np.load(edge_file, mmap_mode="c"))

    masks = []
    for name in ("train_idx.npy", "valid_idx.npy", "test_idx.npy"):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[torch.from_numpy(np.load(npy_dir / name))] = True
        masks.append(mask)
    train_mask, val_mask, test_mask = masks

    return Data(
        x=x, y=y, edge_index=edge_index,
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
        num_nodes=num_nodes,
    )


def parse_size_gb(text: str) -> float:
    """'16GB' / '512MB' / '1TB' / '16' (GB) -> GB."""
    s = text.strip().upper().replace(" ", "")
    for suffix, scale in (("TB", 1024.0), ("GB", 1.0), ("MB", 1 / 1024.0), ("G", 1.0), ("M", 1 / 1024.0)):
        if s.endswith(suffix):
            return float(s[: -len(suffix)]) * scale
    return float(s)


def build_model(args, graph):
    if args.model == "gcn":
        return GCN(
            in_channels=graph.feat_dim,
            hidden_channels=args.hidden,
            out_channels=graph.num_classes,
            num_layers=args.num_layers,
            dropout=args.dropout,
            norm=True,
        )
    if args.model == "gat":
        return GAT(
            in_channels=graph.feat_dim,
            hidden_channels=args.hidden,
            out_channels=graph.num_classes,
            num_layers=args.num_layers,
            heads=args.heads,
            dropout=args.dropout,
            norm=True,
        )
    raise ValueError(f"Unsupported model: {args.model}")


def main():
    parser = argparse.ArgumentParser(description="Train GCN or GAT on ogbn-papers100M with GriNNder")
    parser.add_argument("--ogb_root", type=str, default="data/ogb_datasets",
                        help="Path to OGB datasets directory")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True,
                        help="Download ogbn-papers100M if missing")
    parser.add_argument("--confirm_download", action="store_true",
                        help="Skip OGB's interactive prompt for the large download")
    parser.add_argument("--num_classes", type=int, default=PAPERS_NUM_CLASSES, choices=[PAPERS_NUM_CLASSES])
    parser.add_argument("--undirected", action=argparse.BooleanOptionalAction, default=True,
                        help="Train on the undirected graph (standard for papers100M). "
                             "--no-undirected uses the raw directed citations.")
    parser.add_argument("--preprocess_only", action="store_true",
                        help="Prepare data and partition (saving the partition cache), then exit "
                             "without training. Run where enough RAM is available.")
    parser.add_argument("--mode", type=str, default="grinnder", choices=["hongtu", "grinnder"])
    parser.add_argument("--num_parts", type=int, default=128)
    parser.add_argument("--cache_mode", type=str, default="partition_lru",
                        choices=["auto", "lru_layer", "partition_lru"])
    parser.add_argument("--model", type=str, default="gcn", choices=["gcn", "gat"])
    parser.add_argument("--hidden", type=int, default=128,
                        help="Hidden size. 128 keeps a base epoch at ~360GB of SSD; 256 needs ~590GB.")
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=1, help="GAT attention heads")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--storage_dir", type=str, default="/mnt/nvme")
    parser.add_argument("--runtime_safety_margin_gb", type=float, default=8.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--pool_size", type=int, default=2)
    parser.add_argument("--partition_cache_dir", type=str, default="data/grinnder_partitions")
    parser.add_argument("--no_save_partition_cache", action="store_true")
    parser.add_argument("--partitioner_threads", type=int, default=get_default_partitioner_threads())
    parser.add_argument("--preprocess_workers", type=int, default=get_default_partitioner_threads())
    parser.add_argument("--materialize_features", action="store_true",
                        help="Load the full feature matrix into RAM instead of using mmap")
    
    parser.add_argument("--io_mode", "--io-mode", dest="io_mode", type=str, default="ours",
                        choices=["base", "ours"],
                        help="ours: host FTL, targeted rows SSD<->GPU. "
                             "base: original GriNNder, partition files + DRAM cache.")
    parser.add_argument("--cache-size", "--cache_size", dest="cache_size", type=str, default=None,
                        help="Base only: DRAM limit for the partition cache, e.g. 16GB, 512MB, 16 "
                             "(GB). Default: available memory minus safety margin.")
    parser.add_argument("--breakdown_json", type=str, default="time_breakdown.json",
                        help="Where to save the per-epoch time breakdown (SSD / data movement / compute).")
    args = parser.parse_args()
    if args.preprocess_only and (args.no_save_partition_cache or not args.partition_cache_dir):
        parser.error("--preprocess_only needs the partition cache to be saved")
    cache_size_gb = parse_size_gb(args.cache_size) if args.cache_size else 0.0
    if args.io_mode == "base" and args.cache_mode != "partition_lru":
        print(f"io_mode=base uses cache_mode=partition_lru (was {args.cache_mode})")
        args.cache_mode = "partition_lru"

    fix_seed(args.seed)
    device = args.device

    # ---- Load dataset ----
    print(f"\n\nLoading ogbn-papers100M dataset from {args.ogb_root}...")
    t0 = time.time()
    data = load_papers100m(
        args.ogb_root,
        mmap_features=not args.materialize_features,
        download=args.download,
        confirm_download=args.confirm_download,
        undirected=args.undirected,
    )
    t_load = time.time() - t0
    print(f"  Loaded in {t_load:.1f}s")
    print(f"  Nodes: {data.num_nodes:,}, Edges: {data.edge_index.size(1):,}, "
          f"Features: {data.x.size(1)}, Classes: {args.num_classes}")
    print(f"  Train: {data.train_mask.sum():,}, Val: {data.val_mask.sum():,}, "
          f"Test: {data.test_mask.sum():,}")
    print(f"  {report_memory('After load')}")

    # ---- Configure ----
    config = GriNNderConfig(
        mode=args.mode,
        num_parts=args.num_parts,
        partitioner="grinnder",
        partitioner_kwargs={
            "capacity": 1.1,
            "beta": 1.0,
            "max_iter": 50,
            "num_threads": args.partitioner_threads,
        },
        preprocess_workers=args.preprocess_workers,
        storage_dir=args.storage_dir,
        cache_mode=args.cache_mode,
        runtime_safety_margin_gb=args.runtime_safety_margin_gb,
        pool_size=args.pool_size,
        device=device,
        io_mode=args.io_mode,
        cache_size_gb=cache_size_gb,
    )
    print(
        f"\nIO mode: {config.io_mode}"
        + (f", Cache size: {cache_size_gb:.2f}GB" if config.io_mode == "base" and cache_size_gb else "")
    )
    print(
        f"\nMode: {config.mode}, Partitions: {config.num_parts}, "
        f"Cache: {args.cache_mode}, Partitioner threads: {args.partitioner_threads}, "
        f"Preprocess workers: {args.preprocess_workers}, "
        f"Pool size: {args.pool_size}, "
        f"Safety margin: {args.runtime_safety_margin_gb:.1f}GB"
    )

    # ---- Partition graph ----
    print("\n\n\nPartitioning...")
    cache_path = None
    if args.partition_cache_dir:
        cache_path = (
            Path(args.partition_cache_dir)
            / f"ogbn_papers100m_{'undirected' if args.undirected else 'directed'}"
              f"_grinnder_{args.num_parts}p.pt"
        )
    t0 = time.time()
    graph = build_partitioned_graph_metis(
        edge_index=data.edge_index,
        x=data.x,
        y=data.y,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask,
        config=config,
        cache_path=cache_path,
        save_cache=not args.no_save_partition_cache,
    )

    

    t_part = time.time() - t0
    print(f"  Partitioning: {t_part:.1f}s")
    print(f"  Partition sizes: min={min(graph.partition_sizes):,}, "
          f"max={max(graph.partition_sizes):,}, "
          f"avg={sum(graph.partition_sizes)//len(graph.partition_sizes):,}")
    print(f"  {report_memory('After partition')}\n\n")

    print("\n" + "=" * 80)
    print("      PRUNED PARTITION DEPENDENCY ANALYSIS (FROM GRAPH)")
    print("=" * 80)

    for pid in range(graph.num_parts):
        boundaries = graph.boundaries[pid]
        
        active_deps = []
        if boundaries is not None:
            for src_pid, b in enumerate(boundaries):
                if b is not None and b.numel() > 0:
                    active_deps.append((src_pid, b.numel()))
        
        # Sort by node count descending
        active_deps.sort(key=lambda item: item[1], reverse=True)
        total_ext_nodes = sum(count for _, count in active_deps)
        
        print(
            f"PARTITION {pid:02d} | Own Nodes: {graph.partition_sizes[pid]:,d} "
            f"| Total Ext Boundary Nodes Required: {total_ext_nodes:,d}"
        )
        print(f"Connected to {len(active_deps)}/{graph.num_parts - 1} partitions.")
        print("-" * 80)
        
        dep_strings = []
        for src_pid, count in active_deps:
            pct = (count / total_ext_nodes * 100) if total_ext_nodes > 0 else 0.0
            dep_strings.append(f"P{src_pid:02d}: {count:6,d} ({pct:5.1f}%)")
        
        for i in range(0, len(dep_strings), 4):
            print("   " + "  |  ".join(dep_strings[i:i+4]))
            
        print("=" * 80)

    # ---- SSD space per epoch ----
    # base: input features + every layer output + gradients of the hidden layers.
    # ours: gradients are stored per source partition including its boundary
    #       rows, so they scale with the overlap ratio r = sum(expanded) / N.
    n_nodes = graph.num_nodes
    r = sum(graph.expanded_sizes) / n_nodes
    out_dims = [args.hidden] * (args.num_layers - 1) + [graph.num_classes]
    hidden_dims = out_dims[:-1]
    to_gb = lambda nbytes: nbytes / 1e9
    act_bytes = 4 * n_nodes * (graph.feat_dim + sum(out_dims))
    base_bytes = act_bytes + 4 * n_nodes * sum(hidden_dims)
    ours_bytes = act_bytes + 4 * sum(graph.expanded_sizes) * sum(hidden_dims)
    print(
        f"\nOverlap ratio r = {r:.2f} (expanded partition rows / nodes)\n"
        f"SSD needed per epoch (hidden {args.hidden}, {args.num_layers} layers): "
        f"base ~{to_gb(base_bytes):.0f} GB, ours ~{to_gb(ours_bytes):.0f} GB"
    )

    if args.preprocess_only:
        print(f"\nPreprocessing done. Partition cache: {cache_path}")
        return

    # Free original metadata container; graph keeps an mmap-backed feature source.
    del data
    torch.cuda.empty_cache()

    # ---- Create model ----
    model = build_model(args, graph).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"\n\nModel: {args.model.upper()} "
        f"({args.num_layers} layers, {args.hidden} hidden, {n_params:,} params) \n\n"
    )

    # ---- Train ----
    trainer = Trainer(model=model, graph=graph, config=config)
    if trainer.cache is not None:
        plan = trainer.cache.memory_plan
        to_gb = lambda nbytes: nbytes / float(1024**3)
        print(
            "Cache plan: \n"
            f"mode={trainer.cache.mode}, \n"
            f"remaining={to_gb(plan.remaining_cache_bytes):.2f}GB, \n"
            f"activation_budget={to_gb(trainer.cache.activation_cache_budget_bytes):.2f}GB, \n"
            f"all_layer_residency={to_gb(plan.all_layer_residency_bytes):.2f}GB, \n"
            f"layer_working_set={to_gb(plan.layer_working_set_bytes):.2f}GB, \n"
            f"safety_margin={to_gb(plan.safety_margin_bytes):.2f}GB\n\n"
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    print(f"\nTraining for {args.epochs} epochs...\n")
    print(f"{'Epoch':>6} | {'Loss':>8} | {'Val Acc':>8} | {'Test Acc':>9} | {'Time':>8}")
    print("-" * 56)

    best_val = 0.0
    best_test = 0.0
    epoch_times = []
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        metrics = trainer.train_epoch(optimizer, criterion)
        t_epoch = time.time() - t0
        epoch_times.append(t_epoch)

        stat.print_timeline()
        stat.breakdown.print_epoch(title=f"epoch {epoch}")
        print(f"Epoch Time = {t_epoch}")
        

        if metrics["val_acc"] > best_val:
            best_val = metrics["val_acc"]
            best_test = metrics["test_acc"]

        print(
            f"{epoch:6d} | {metrics['loss']:8.4f} | {metrics['val_acc']:7.2%} | "
            f"{metrics['test_acc']:8.2%} | {t_epoch:7.1f}s"
        )

    t_total = time.time() - t_start

    print("-" * 56)
    print(
        f"\nTraining complete in {t_total:.1f}s "
        f"({statistics.median(epoch_times):.1f}s median epoch)"
    )
    print(f"Best Val: {best_val:.2%} | Best Test: {best_test:.2%}")
    print(f"\n{report_memory('Final')}")

    stat.breakdown.print_summary()
    stat.breakdown.save(args.breakdown_json)
    print(f"Time breakdown saved to {args.breakdown_json}")


if __name__ == "__main__":
    main()
