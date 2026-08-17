"""GriNNder: Breaking the Memory Capacity Wall in Full-Graph GNN Training with Storage Offloading."""

__version__ = "0.1.0"

# Lazy imports to avoid requiring torch_geometric at import time.
# This allows `from grinnder.config import GriNNderConfig` to work
# without torch_geometric installed.


def __getattr__(name):
    if name == "GriNNderConfig":
        from grinnder.config import GriNNderConfig
        return GriNNderConfig
    if name == "GriNNderModel":
        from grinnder.nn.base import GriNNderModel
        return GriNNderModel
    if name == "GCN":
        from grinnder.nn.gcn import GCN
        return GCN
    if name == "GAT":
        from grinnder.nn.gat import GAT
        return GAT
    if name == "Trainer":
        from grinnder.engine.trainer import Trainer
        return Trainer
    if name == "PartitionedGraph":
        from grinnder.data.graph import PartitionedGraph
        return PartitionedGraph
    if name == "build_partitioned_graph":
        from grinnder.data.partition import build_partitioned_graph
        return build_partitioned_graph
    if name == "build_partitioned_graph_metis":
        from grinnder.data.partition import build_partitioned_graph_metis
        return build_partitioned_graph_metis
    if name == "load_dataset":
        from grinnder.data.datasets import load_dataset
        return load_dataset
    if name == "load_igb":
        from grinnder.data.datasets import load_igb
        return load_igb
    if name == "load_papers100M":
        from grinnder.data.datasets import load_papers100M
        return load_papers100M
    raise AttributeError(f"module 'grinnder' has no attribute {name!r}")


__all__ = [
    "GriNNderConfig",
    "GriNNderModel",
    "GCN",
    "GAT",
    "Trainer",
    "PartitionedGraph",
    "build_partitioned_graph",
    "build_partitioned_graph_metis",
    "load_dataset",
    "load_igb",
    "load_papers100M",
]
