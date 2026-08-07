"""Train GCN or GAT on IGB-Medium (10M nodes, 120M edges) with GriNNder.

This is the real-world benchmark from the paper. With 10M nodes and 1024-dim
features, the intermediate activations exceed GPU memory, requiring GriNNder's
storage offloading.

Usage:
  python examples/train_igb.py --igb_root data/igb_datasets
  python examples/train_igb.py --igb_root data/igb_datasets --mode hongtu
  python examples/train_igb.py --igb_root data/igb_datasets --cache_mode lru_layer --storage_dir /pci5_nvme/grinnder
"""

import argparse
import statistics
import time

import torch

from grinnder import GAT, GCN, GriNNderConfig, Trainer, build_partitioned_graph
from grinnder.data.datasets import load_igb
from grinnder.utils import fix_seed, get_default_partitioner_threads, report_memory


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
    parser = argparse.ArgumentParser(description="Train GCN or GAT on IGB-Medium with GriNNder")
    parser.add_argument("--igb_root", type=str, default="data/igb_datasets",
                        help="Path to igb_datasets directory")
    parser.add_argument("--igb_size", type=str, default="medium",
                        choices=["tiny", "small", "medium", "large"])
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True,
                        help="Download homogeneous IGB tiny/small/medium if missing")
    parser.add_argument("--confirm_download", action="store_true",
                        help="Skip the interactive prompt for large IGB downloads")
    parser.add_argument("--num_classes", type=int, default=19, choices=[19, 2983])
    parser.add_argument("--mode", type=str, default="grinnder", choices=["hongtu", "grinnder"])
    parser.add_argument("--num_parts", type=int, default=64)
    parser.add_argument("--cache_mode", type=str, default="partition_lru",
                        choices=["auto", "lru_layer", "partition_lru"])
    parser.add_argument("--model", type=str, default="gcn", choices=["gcn", "gat"])
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=1, help="GAT attention heads")
    parser.add_argument("--epochs", type=int, default=10)
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
                        help="Load the full IGB feature matrix into RAM instead of using mmap")
    args = parser.parse_args()

    fix_seed(args.seed)
    device = args.device

    # ---- Load dataset ----
    print(f"\n\nLoading IGB-{args.igb_size} dataset from {args.igb_root}...")
    t0 = time.time()
    data = load_igb(
        args.igb_root,
        size=args.igb_size,
        num_classes=args.num_classes,
        mmap_features=not args.materialize_features,
        download=args.download,
        confirm_download=args.confirm_download,
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
        from pathlib import Path

        cache_path = (
            Path(args.partition_cache_dir)
            / f"igb_{args.igb_size}_grinnder_{args.num_parts}p.pt"
        )
    t0 = time.time()
    graph = build_partitioned_graph(
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

    print("\n" + "="*80)
    print("      DETAILED METIS PARTITION DEPENDENCY ANALYSIS")
    print("="*80)
    
    print(f"Partitioning graph into {args.num_parts} parts...")
    t_metis_start = time.time()
    
    cluster_data = ClusterData(
        data, 
        num_parts=args.num_parts, 
        recursive=False, 
        save_dir="/tmp/metis_test"
    )
    print(f"METIS finished in {time.time() - t_metis_start:.1f}s\n")
    
    # Safely retrieve partition mapping
    partition = cluster_data.partition
    partptr, perm = None, None
    for attr in ["partptr", "partition_ptr", "node_perm_ptr", "_ptr", "ptr"]:
        if hasattr(partition, attr):
            partptr = getattr(partition, attr)
            break
            
    for attr in ["node_perm", "perm", "_perm"]:
        if hasattr(partition, attr):
            perm = getattr(partition, attr)
            break

    # Reconstruct node -> partition mapping
    num_nodes = data.num_nodes
    node_to_part = torch.empty(num_nodes, dtype=torch.long)
    
    if partptr is not None and perm is not None:
        for p_id in range(args.num_parts):
            start, end = partptr[p_id], partptr[p_id + 1]
            nodes_in_part = perm[start:end]
            node_to_part[nodes_in_part] = p_id
    else:
        for p_id in range(args.num_parts):
            part_data = cluster_data[p_id]
            node_ids = part_data.n_id if hasattr(part_data, "n_id") else part_data.input_id
            node_to_part[node_ids] = p_id

    # Compute boundary node counts across partitions
    src, dst = data.edge_index[0], data.edge_index[1]
    src_part = node_to_part[src]
    dst_part = node_to_part[dst]

    # Filter cross-partition edges
    cross_mask = src_part != dst_part
    cross_src_part = src_part[cross_mask]
    cross_dst_part = dst_part[cross_mask]
    cross_dst_nodes = dst[cross_mask]

    # Build dependency matrix: size_matrix[target_part, source_part]
    dep_matrix = torch.zeros((args.num_parts, args.num_parts), dtype=torch.long)

    for p_id in range(args.num_parts):
        p_mask = cross_dst_part == p_id
        if not p_mask.any():
            continue
        
        p_src_parts = cross_src_part[p_mask]
        p_dst_nodes = cross_dst_nodes[p_mask]
        
        for neighbor_pid in range(args.num_parts):
            if p_id == neighbor_pid:
                continue
            n_mask = p_src_parts == neighbor_pid
            if n_mask.any():
                # Unique nodes needed from neighbor_pid by partition p_id
                unique_nodes = torch.unique(p_dst_nodes[n_mask]).numel()
                dep_matrix[p_id, neighbor_pid] = unique_nodes

    # Format and display output per partition
    print("="*80)
    for pid in range(args.num_parts):
        row = dep_matrix[pid]
        total_ext_nodes = row.sum().item()
        
        # Get non-zero dependencies
        valid_indices = (row > 0).nonzero(as_tuple=True)[0]
        valid_counts = row[valid_indices]
        
        # Sort dependencies in descending order
        sorted_counts, sort_order = torch.sort(valid_counts, descending=True)
        sorted_neighbors = valid_indices[sort_order]
        
        num_neighbors = len(sorted_counts)
        print(f"PARTITION {pid:02d} | Own Nodes: {(node_to_part == pid).sum().item():,d} | Total Ext Boundary Nodes Required: {total_ext_nodes:,d}")
        print(f"Connected to {num_neighbors}/{args.num_parts-1} partitions.")
        print("-" * 80)
        
        # List exact counts and percentages for each dependent neighbor partition
        dep_strings = []
        for neighbor_pid, count in zip(sorted_neighbors, sorted_counts):
            c_val = count.item()
            pct = (c_val / total_ext_nodes * 100) if total_ext_nodes > 0 else 0.0
            dep_strings.append(f"P{neighbor_pid.item():02d}: {c_val:6,d} ({pct:5.1f}%)")
        
        # Print in formatted rows of 4 dependencies per line for readability
        for i in range(0, len(dep_strings), 4):
            print("   " + "  |  ".join(dep_strings[i:i+4]))
            
        print("="*80)

    exit(1)

    '''
    # ---- Fast METIS Diagnostic Test ----
    print("\n" + "="*60)
    print("      RUNNING FAST METIS DEPENDENCY DIAGNOSTIC TEST")
    print("="*60)
    
    from torch_geometric.loader import ClusterData
    
    print(f"Partitioning graph with METIS into {args.num_parts} parts...")
    t_metis_start = time.time()
    
    # Run METIS partitioning
    cluster_data = ClusterData(
        data, 
        num_parts=args.num_parts, 
        recursive=False, 
        save_dir="/tmp/metis_test"
    )
    print(f"METIS finished in {time.time() - t_metis_start:.1f}s")
    
    # Safely retrieve partition mapping
    partition = cluster_data.partition
    
    # Find partptr attribute
    partptr = None
    for attr in ["partptr", "partition_ptr", "node_perm_ptr", "_ptr", "ptr"]:
        if hasattr(partition, attr):
            partptr = getattr(partition, attr)
            break
            
    # Find perm attribute
    perm = None
    for attr in ["node_perm", "perm", "_perm"]:
        if hasattr(partition, attr):
            perm = getattr(partition, attr)
            break

    # Direct fallback for PyG ClusterData internals
    if partptr is None or perm is None:
        # Reconstruct node_to_part directly using ClusterData's __getitem__ or internal tensor
        node_to_part = torch.empty(data.num_nodes, dtype=torch.long)
        for p_id in range(args.num_parts):
            part_data = cluster_data[p_id]
            if hasattr(part_data, "n_id"):
                node_to_part[part_data.n_id] = p_id
            elif hasattr(part_data, "input_id"):
                node_to_part[part_data.input_id] = p_id
    else:
        # Build node_id -> partition_id mapping from pointers
        num_nodes = data.num_nodes
        node_to_part = torch.empty(num_nodes, dtype=torch.long)
        for p_id in range(args.num_parts):
            start, end = partptr[p_id], partptr[p_id + 1]
            nodes_in_part = perm[start:end]
            node_to_part[nodes_in_part] = p_id

    # Compute boundary node counts across partitions
    src, dst = data.edge_index[0], data.edge_index[1]
    src_part = node_to_part[src]
    dst_part = node_to_part[dst]

    # Filter cross-partition boundary edges
    cross_mask = src_part != dst_part
    cross_src_part = src_part[cross_mask]
    cross_dst_part = dst_part[cross_mask]
    cross_dst_nodes = dst[cross_mask]

    metis_size_matrix = torch.zeros((args.num_parts, args.num_parts), dtype=torch.long)

    # Count unique target nodes requested by target partition from source partition
    for p_id in range(args.num_parts):
        p_mask = cross_dst_part == p_id
        if not p_mask.any():
            continue
        
        p_src_parts = cross_src_part[p_mask]
        p_dst_nodes = cross_dst_nodes[p_mask]
        
        for neighbor_pid in range(args.num_parts):
            if p_id == neighbor_pid:
                continue
            n_mask = p_src_parts == neighbor_pid
            if n_mask.any():
                # Count unique required boundary nodes from neighbor_pid
                unique_nodes = torch.unique(p_dst_nodes[n_mask]).numel()
                metis_size_matrix[p_id, neighbor_pid] = unique_nodes

    print("\n" + "="*60)
    print("      METIS PARTITION DEPENDENCY DISTRIBUTION")
    print("="*60)

    total_dense_pairs = 0
    total_possible_pairs = args.num_parts * (args.num_parts - 1)

    for pid in range(args.num_parts):
        counts = metis_size_matrix[pid].clone().float()
        counts[pid] = 0  # Ignore self
        
        connected = (counts > 0).sum().item()
        total_dense_pairs += connected

        valid_counts = counts[counts > 0].sort(descending=True).values
        total_boundary_nodes = valid_counts.sum().item()

        if len(valid_counts) > 0:
            top_20_percent_k = max(1, int(len(valid_counts) * 0.2))
            top_nodes = valid_counts[:top_20_percent_k].sum().item()
            ratio = (top_nodes / total_boundary_nodes * 100) if total_boundary_nodes > 0 else 0
            
            max_nodes = int(valid_counts[0].item())
            min_nodes = int(valid_counts[-1].item())

            print(
                f"Partition {pid:02d} | Connected to {connected:02d}/{args.num_parts-1} parts | "
                f"Top 20% parts account for {ratio:5.1f}% of nodes | "
                f"Max boundary: {max_nodes:6d} | Min boundary: {min_nodes:4d}"
            )

    overall_density = (total_dense_pairs / total_possible_pairs) * 100
    print("-" * 60)
    print(f"METIS Boolean Dependency Matrix Density: {overall_density:.2f}%")
    print("=" * 60 + "\n")

    import sys
    print("Fast METIS test complete. Exiting...")
    sys.exit(0)
    '''

    

    t_part = time.time() - t0
    print(f"  Partitioning: {t_part:.1f}s")
    print(f"  Partition sizes: min={min(graph.partition_sizes):,}, "
          f"max={max(graph.partition_sizes):,}, "
          f"avg={sum(graph.partition_sizes)//len(graph.partition_sizes):,}")
    print(f"  {report_memory('After partition')}\n\n")

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
