"""Train GCN or GAT on ogbn-papers100M (111M nodes, 1.61B edges) with GriNNder.

This script mirrors the IGB training benchmark for the massive ogbn-papers100M dataset.
Because of the dataset's scale, intermediate activations easily exceed VRAM, requiring
GriNNder's partitioning and NVMe storage offloading.

Usage:
  python examples/train_ogb_papers100m.py --ogb_root data/ogb_datasets
  python examples/train_ogb_papers100m.py --ogb_root data/ogb_datasets --mode hongtu
  python examples/train_ogb_papers100m.py --ogb_root data/ogb_datasets --cache_mode lru_layer --storage_dir /mnt/nvme
"""

import argparse
import statistics
import time
from pathlib import Path
from typing import Any, NamedTuple

import torch

from ogb.nodeproppred import PygNodePropPredDataset

from grinnder import (
    GAT,
    GCN,
    GriNNderConfig,
    Trainer,
    build_partitioned_graph,
    build_partitioned_graph_metis,
)
from grinnder.stats import stat
from grinnder.utils import fix_seed, get_default_partitioner_threads, report_memory


class OGBDataContainer(NamedTuple):
    """Container holding ogbn-papers100M attributes structured identically to PyG / IGB loader objects."""

    num_nodes: int
    edge_index: torch.Tensor
    x: Any
    y: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor


def load_ogb_papers100m(root: str) -> OGBDataContainer:
    """Downloads (if necessary) and loads the ogbn-papers100M dataset.

    Converts OGB node split indices into boolean mask tensors matching GriNNder expected types.
    """
    print(f"Downloading/Loading ogbn-papers100M dataset from {root}...")
    dataset = PygNodePropPredDataset(name="ogbn-papers100M", root=root)
    data = dataset[0]

    num_nodes = data.num_nodes

    # Convert node indices from OGB format to boolean masks
    split_idx = dataset.get_idx_split()
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[split_idx["train"]] = True
    val_mask[split_idx["valid"]] = True
    test_mask[split_idx["test"]] = True

    # Flatten label tensor to shape (num_nodes,) with long type
    y = data.y.squeeze(-1)
    if y.dtype != torch.long:
        y = y.to(torch.long)

    return OGBDataContainer(
        num_nodes=num_nodes,
        edge_index=data.edge_index,
        x=data.x,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )


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
    parser = argparse.ArgumentParser(
        description="Train GCN or GAT on ogbn-papers100M with GriNNder"
    )
    parser.add_argument(
        "--ogb_root",
        type=str,
        default="data/ogb_datasets",
        help="Path to ogb_datasets directory",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=172,
        help="Number of classes in ogbn-papers100M",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="grinnder",
        choices=["hongtu", "grinnder"],
    )
    parser.add_argument("--num_parts", type=int, default=128)
    parser.add_argument(
        "--cache_mode",
        type=str,
        default="partition_lru",
        choices=["auto", "lru_layer", "partition_lru"],
    )
    parser.add_argument(
        "--model", type=str, default="gcn", choices=["gcn", "gat"]
    )
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument(
        "--heads", type=int, default=1, help="GAT attention heads"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--storage_dir", type=str, default="/mnt/nvme")
    parser.add_argument("--runtime_safety_margin_gb", type=float, default=8.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--pool_size", type=int, default=2)
    parser.add_argument(
        "--partition_cache_dir",
        type=str,
        default="data/grinnder_partitions",
    )
    parser.add_argument("--no_save_partition_cache", action="store_true")
    parser.add_argument(
        "--partitioner_threads",
        type=int,
        default=get_default_partitioner_threads(),
    )
    parser.add_argument(
        "--preprocess_workers",
        type=int,
        default=get_default_partitioner_threads(),
    )
    parser.add_argument(
        "--sparse",
        type=int,
        default=0,
        help="Sparsify graph to reduce partition dependencies.",
    )
    args = parser.parse_args()

    fix_seed(args.seed)
    device = args.device
    sp = args.sparse

    # ---- Load dataset ----
    print(f"\n\nLoading ogbn-papers100M dataset from {args.ogb_root}...")
    t0 = time.time()
    data = load_ogb_papers100m(args.ogb_root)
    t_load = time.time() - t0
    print(f"  Loaded in {t_load:.1f}s")
    print(
        f"  Nodes: {data.num_nodes:,}, Edges: {data.edge_index.size(1):,}, "
        f"Features: {data.x.size(1)}, Classes: {args.num_classes}"
    )
    print(
        f"  Train: {data.train_mask.sum():,}, Val: {data.val_mask.sum():,}, "
        f"Test: {data.test_mask.sum():,}"
    )
    print(f"  {report_memory('After load')}")

    # ---- Configure ----
    config = GriNNderConfig(
        mode=args.mode,
        num_parts=args.num_parts,
        partitioner="metis",
        partitioner_kwargs={
            "recursive": False,
            "num_threads": args.partitioner_threads,
        },
        preprocess_workers=args.preprocess_workers,
        storage_dir=args.storage_dir,
        cache_mode=args.cache_mode,
        runtime_safety_margin_gb=args.runtime_safety_margin_gb,
        pool_size=args.pool_size,
        device=device,
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
        Path(args.partition_cache_dir).mkdir(parents=True, exist_ok=True)
        cache_path = (
            Path(args.partition_cache_dir)
            / f"ogb_papers100m_metis_{args.num_parts}p.pt"
        )
    t0 = time.time()

    # Pass edge_index and features directly matching your build_partitioned_graph_metis signature
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
    print(
        f"  Partition sizes: min={min(graph.partition_sizes):,}, "
        f"max={max(graph.partition_sizes):,}, "
        f"avg={sum(graph.partition_sizes)//len(graph.partition_sizes):,}"
    )
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
        print(
            f"Connected to {len(active_deps)}/{graph.num_parts - 1} partitions."
        )
        print("-" * 80)

        dep_strings = []
        for src_pid, count in active_deps:
            pct = (
                (count / total_ext_nodes * 100) if total_ext_nodes > 0 else 0.0
            )
            dep_strings.append(f"P{src_pid:02d}: {count:6,d} ({pct:5.1f}%)")

        for i in range(0, len(dep_strings), 4):
            print("   " + "  |  ".join(dep_strings[i : i + 4]))

        print("=" * 80)

    # Free original dataset structure; memory maps / graph tensors are stored in PartitionedGraph
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
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=5e-4
    )
    criterion = torch.nn.CrossEntropyLoss()

    print(f"\nTraining for {args.epochs} epochs...\n")
    print(
        f"{'Epoch':>6} | {'Loss':>8} | {'Val Acc':>8} | {'Test Acc':>9} | {'Time':>8}"
    )
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


if __name__ == "__main__":
    main()