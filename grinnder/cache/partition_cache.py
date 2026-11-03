"""Partition-granularity caching for host memory.

Manages which layers' activations are kept in host RAM vs NVMe storage.
Auto-selection uses the demand-managed modes:

  Mode 1 (lru_layer):     Whole-layer LRU, usually all-resident on large hosts
  Mode 2 (partition_lru): Per-partition LRU eviction (extreme scarcity)
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Set, Tuple

import torch

from grinnder.buffer.host import HostBuffer
from grinnder.storage.backend import StorageBackend
from grinnder.utils import get_available_host_memory

@dataclass(frozen=False)
class PartitionCacheTracker:
    """Track partition and gradient cache status"""
    partitions: List[List[int],List[int],List[int]]
    gradients: List[List[int],List[int],List[int]]
    num_parts: int
    num_grads: int

    def add_partition(self, layer, pid, size):
        if self.partitions[layer][pid] == 0:
            self.partitions[layer][pid] = round(size/(1024**3),2)
            self.num_parts += 1

    def remove_partition(self, layer, pid):
        if self.partitions[layer][pid] != 0:
            self.partitions[layer][pid] = 0
            self.num_parts -= 1

    def add_gradient(self, layer, pid, size):
        if self.gradients[layer][pid] == 0:
            self.gradients[layer][pid] = round(size/(1024**3),2)
            self.num_grads += 1

    def remove_gradient(self, layer, pid):
        if self.gradients[layer][pid] != 0:
            self.gradients[layer][pid] = 0
            self.num_grads -= 1

    def print_cache(self):
        total_part = sum(self.partitions[0]) + sum(self.partitions[1]) + sum(self.partitions[2])
        total_part = round(total_part,2)

        total_grad = sum(self.gradients[0]) + sum(self.gradients[1]) + sum(self.gradients[2])
        total_grad = round(total_grad,2)


        print('\n*******************')
        print(f'[Cache] : Partitions[{total_part}GB] + Gradients[{total_grad}GB] = [{total_part+total_grad}GB]')
        print(f'\t[Partitions] [{self.num_parts}]')
        print('---------------------------------')
        print(f'|{self.partitions[0]}|\n|{self.partitions[1]}|\n|{self.partitions[2]}|')
        print('---------------------------------')
        print(f'\t[Gradients] [{self.num_grads}]')
        print('---------------------------------')
        print(f'|{self.gradients[0]}|\n|{self.gradients[1]}|\n|{self.gradients[2]}|')
        print('---------------------------------')



@dataclass(frozen=True)
class CacheMemoryPlan:
    """Concrete host-memory accounting used for cache-mode selection."""

    total_budget_bytes: int
    fixed_resident_bytes: int
    safety_margin_bytes: int
    remaining_cache_bytes: int
    activation_layer_bytes: Tuple[int, ...]
    gradient_layer_bytes: Tuple[int, ...]
    all_layer_residency_bytes: int
    layer_working_set_bytes: int
    largest_partition_dependency_bytes: int


class PartitionCache:
    """Partition-granularity caching with hierarchical replacement.

    Manages host memory for activations (forward gather, backward regather)
    and gradient write-back buffers. Auto-detects the best cache mode
    based on available host memory.

    Args:
        num_parts: Number of partitions.
        num_layers: Number of GNN layers.
        layer_dims: [in_dim, hid_dim, ..., out_dim] (len = num_layers + 1).
        num_nodes: Total number of nodes in graph.
        host_buffers: List of HostBuffers, one per layer (index 0..L).
        grad_buffers: List of HostBuffers for gradients (index 1..L-1, None for layer 0).
        backend: StorageBackend for NVMe I/O.
        mode: Cache mode or 'auto' for auto-detection.
        host_memory_budget_gb: Budget in GB (0 = auto-detect).
    """

    def __init__(
        self,
        num_parts: int,
        num_layers: int,
        layer_dims: List[int],
        num_nodes: int,
        host_buffers: List[HostBuffer],
        grad_buffers: List[Optional[HostBuffer]],
        backend: StorageBackend,
        mode: Literal["auto", "lru_layer", "partition_lru"] = "partition_lru",
        host_memory_budget_gb: float = 0,
        fixed_resident_bytes: int = 0,
        safety_margin_bytes: int = 0,
        dependency_sets: Optional[List[Set[int]]] = None,
        cache_size_bytes: int = 0,
    ):
        valid_modes = {"auto", "lru_layer", "partition_lru"}
        if mode not in valid_modes:
            raise ValueError(
                f"Unsupported cache mode {mode!r}; expected one of "
                f"{sorted(valid_modes)}"
            )

        self.num_parts = num_parts
        self.num_layers = num_layers
        self.layer_dims = layer_dims
        self.num_nodes = num_nodes
        self.host_buffers = host_buffers
        self.grad_buffers = grad_buffers
        self.backend = backend
        self.dependency_sets = dependency_sets

        if host_memory_budget_gb > 0:
            self._budget = int(host_memory_budget_gb * (1024**3))
        else:
            self._budget = get_available_host_memory()
        self._fixed_resident_bytes = max(0, int(fixed_resident_bytes))
        self._safety_margin_bytes = max(0, int(safety_margin_bytes))
        self._memory_plan = self._build_memory_plan()
        self._cache_tracker = self._build_cache_tracker(num_parts)
        self._layer_activation_cache_budget = max(
            0,
            self._memory_plan.remaining_cache_bytes
            - max(self._memory_plan.gradient_layer_bytes or (0,)),
        )
        self._partition_activation_cache_budget = self._memory_plan.remaining_cache_bytes

        if mode == "auto":
            self._mode = self._auto_detect()
        else:
            self._mode = mode
        self._activation_cache_budget = (
            self._layer_activation_cache_budget
            if self._mode == "lru_layer"
            else self._partition_activation_cache_budget
        )
        # --cache-size (io_mode=base): DRAM limit for cached activation AND
        # gradient partitions together, as given.
        self._cache_size_bytes = max(0, int(cache_size_bytes))
        if self._cache_size_bytes > 0:
            self._activation_cache_budget = self._cache_size_bytes
            print(
                f"[cache] DRAM budget for cached partitions (activations + "
                f"gradients): {self._cache_size_bytes / 1024**3:.2f} GB (--cache-size)"
            )

        # Track which layers are currently in host cache (layer-level tracking)
        self._cached_layers: OrderedDict[int, bool] = OrderedDict()

        # partition_lru: one LRU for everything cached in DRAM.
        #   (layer, pid)      activation partition
        #   ("g", layer, pid) gradient partition being accumulated (backward)
        self._cached_partitions: OrderedDict[Tuple, bool] = OrderedDict()

        # Hit/miss counters
        self._hits = 0
        self._misses = 0
        self._resident_activation_bytes = 0
        # Gradient partitions in DRAM being accumulated (in the LRU)
        self._resident_gradient_bytes = 0
        # (layer, pid) gradient partitions evicted mid-layer: their file holds
        # the partial sum, so they are reloaded (not zeroed) when needed again.
        self._spilled_gradients: Set[Tuple[int, int]] = set()
        # ("g"|"a", layer, pid) -> bytes: partitions loaded only to be uploaded
        # to the GPU (next-layer gradients, last-layer activations for the
        # loss). Counted in the budget, never evicted (the copy may be running).
        self._uploads: Dict[Tuple[str, int, int], int] = {}
        self._upload_bytes = 0
        # Activation keys of the last async gather (DRAM -> GPU may still be
        # running): not evicted until the copies are waited for.
        self._inflight_gather_keys: Set[Tuple[int, int]] = set()
        # CPU copies made by the C++ gather (boundary rows, keep_alive) and
        # scatter (prev_values + one boundary block) still running.
        self._inflight_gather_bytes = 0
        self._inflight_scatter_bytes = 0
        # Synchronous temporary CPU copy of the swapping fallbacks
        self._temp_bytes = 0

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def memory_plan(self) -> CacheMemoryPlan:
        return self._memory_plan

    @property
    def remaining_cache_bytes(self) -> int:
        return self._memory_plan.remaining_cache_bytes

    @property
    def activation_cache_budget_bytes(self) -> int:
        """Host bytes available for resident activation cache entries."""
        return self._activation_cache_budget

    def _auto_detect(self) -> str:
        """Select cache mode based on available host memory.

        Host memory model:
          lru_layer:     cache holds full activation layers and evicts whole
                         layers only under budget pressure.
          partition_lru: cache holds dependency partitions and bypasses forward
                         outputs to storage.
          Always:        boundaries + per-partition adj temps.
        """
        available = self._layer_activation_cache_budget

        if available >= self._memory_plan.layer_working_set_bytes:
            return "lru_layer"
        return "partition_lru"

    def _build_memory_plan(self) -> CacheMemoryPlan:
        activation_layer_bytes = tuple(buf.total_nbytes for buf in self.host_buffers)
        gradient_layer_bytes = tuple(
            0 if buf is None else buf.total_nbytes for buf in self.grad_buffers
        )
        all_layer_residency_bytes = sum(activation_layer_bytes) + sum(
            gradient_layer_bytes
        )
        max_activation = max(activation_layer_bytes or (0,))
        max_gradient = max(gradient_layer_bytes or (0,))
        layer_working_set_bytes = max_activation + max_gradient
        largest_dependency = self._largest_partition_dependency_bytes()
        remaining = max(
            0,
            self._budget - self._fixed_resident_bytes - self._safety_margin_bytes,
        )
        return CacheMemoryPlan(
            total_budget_bytes=self._budget,
            fixed_resident_bytes=self._fixed_resident_bytes,
            safety_margin_bytes=self._safety_margin_bytes,
            remaining_cache_bytes=remaining,
            activation_layer_bytes=activation_layer_bytes,
            gradient_layer_bytes=gradient_layer_bytes,
            all_layer_residency_bytes=all_layer_residency_bytes,
            layer_working_set_bytes=layer_working_set_bytes,
            largest_partition_dependency_bytes=largest_dependency,
        )

    def _largest_partition_dependency_bytes(self) -> int:
        largest = 0
        layer_id = self._largest_activation_layer_id()
        for pid in range(self.num_parts):
            deps = self._dependency_set(pid)
            total = sum(self._partition_bytes(layer_id, src_pid) for src_pid in deps)
            largest = max(largest, total)
        return largest

    def _largest_activation_layer_id(self) -> int:
        sizes = [buf.total_nbytes for buf in self.host_buffers]
        return max(range(len(sizes)), key=lambda i: sizes[i]) if sizes else 0

    def ensure_in_host(self, layer_id: int) -> None:
        """Ensure layer's activations are in host cache for gather/regather.

        lru_layer: load full layer from storage, evict LRU layer if needed.
        partition_lru: individual partitions loaded on demand via
               ensure_partition_in_host(). This method pre-loads all partitions
               of the layer, evicting other (layer, partition) pairs as needed.
        """
        if self._mode == "lru_layer":
            if layer_id in self._cached_layers:
                self._cached_layers.move_to_end(layer_id)
                self._hits += 1
                return

            self._misses += 1

            # Evict oldest layer(s) until budget allows new layer
            layer_bytes = self._layer_bytes(layer_id)
            if layer_bytes > self._activation_cache_budget:
                raise MemoryError(
                    f"Layer {layer_id} requires {layer_bytes} bytes, but only "
                    f"{self._activation_cache_budget} activation-cache bytes remain"
                )
            while (
                self._resident_activation_bytes + layer_bytes
                > self._activation_cache_budget
            ):
                if not self._cached_layers:
                    raise MemoryError(
                        f"Cannot make room for layer {layer_id} within "
                        f"{self._activation_cache_budget} bytes"
                    )
                evict_id, _ = self._cached_layers.popitem(last=False)
                self._evict_layer(evict_id)

            # Load full layer from storage only for partitions that are not
            # already resident. Newly produced layers may still be in host RAM.
            for pid in range(self.num_parts):
                if not self.host_buffers[layer_id].is_allocated(pid):
                    self.host_buffers[layer_id].storage_to_cpu(pid=pid)
            self._cached_layers[layer_id] = True
            self._resident_activation_bytes += layer_bytes
            return

        if self._mode == "partition_lru":
            # Compatibility path: callers that do not provide a target partition
            # request the whole layer. The trainer uses dependency-aware calls.
            phase = "none"
            for pid in range(self.num_parts):
                self.ensure_partition_in_host(layer_id, pid, phase)
            return

    def ensure_partition_in_host(self, layer_id: int, pid: int, phase) -> None:
        """Ensure a single (layer, partition) is in host cache.

        Used in partition_lru mode. Evicts least-recently-used (layer, partition)
        pairs when host memory budget is exceeded.
        """
        self.ensure_dependencies_in_host(layer_id, pid, phase, dependencies={pid})

    def ensure_dependencies_in_host(
        self,
        layer_id: int,
        target_pid: int,
        phase,
        boundaries: Optional[List[Optional[torch.Tensor]]] = None,
        dependencies: Optional[Set[int]] = None,
        extra_bytes: int = 0,
    ) -> None:
        """Ensure the dependency partitions for a target partition are resident
        (leaving room for `extra_bytes` more, e.g. the gather's CPU copy).

        Partition-wise replacement is demand-aware: all partitions needed for the
        current target partition's gather/regather are protected from eviction,
        while older unrelated partitions are evicted until the dependency set
        fits the remaining activation-cache budget.
        """
        deps = self._dependency_set(target_pid, boundaries, dependencies)
        keys = {(layer_id, pid) for pid in deps}

        print(f"Partition {target_pid} requires {len(keys)} other partitions")

        demand_bytes = sum(self._partition_bytes(layer_id, pid) for pid in deps)
        if demand_bytes > self._activation_cache_budget:
            raise MemoryError(
                f"\n*****************************************\n"
                f"Layer {layer_id}, partition {target_pid} dependencies require "
                f"{demand_bytes/1024/1024} MB, but only {self._activation_cache_budget/1024/1024} "
                "activation-cache MB remain"
                f"\n*****************************************\n"
            )

        missing = []
        for key in sorted(keys):
            if key in self._cached_partitions:
                self._cached_partitions.move_to_end(key)
                self._hits += 1
            else:
                missing.append(key)
                self._misses += 1

        missing_bytes = sum(self._partition_bytes(layer_id, pid) for _, pid in missing)
        if not self._make_room(missing_bytes + extra_bytes, keys):
            raise MemoryError(
                f"Cannot make room for layer {layer_id}, partition "
                f"{target_pid} dependency set within "
                f"{self._activation_cache_budget} bytes"
            )

        # target partition first
        self._cache_tracker.add_partition(layer_id, target_pid, self._partition_bytes(layer_id, target_pid))

        for key in missing:
            _, pid = key
            self.host_buffers[layer_id].storage_to_cpu(phase, pid=pid)
            self._cached_partitions[key] = True
            self._resident_activation_bytes += self._partition_bytes(layer_id, pid)
            self._cache_tracker.add_partition(layer_id, pid, self._partition_bytes(layer_id, pid))
            print(f"Loaded partition [Layer = {layer_id} | PID = {pid}] | Partition Size = {self._partition_bytes(layer_id, pid)}")
            

        

    def gather_with_swapping(
        self,
        phase,
        layer_id: int,
        target_pid: int,
        gpu_target: torch.Tensor,
        boundaries: List[Optional[torch.Tensor]],
        stream: torch.cuda.Stream,
    ) -> None:
        """Base gather: whole dependency partitions storage -> DRAM, then DRAM -> GPU.

        Dependencies + the gather's CPU copy of the boundary rows fit the budget:
        load them all, one C++ gather (original). Otherwise: go through the
        source partitions one at a time (cached ones first): load it (evicting
        LRU partitions), copy its rows into their fixed GPU row range
        [intra | boundary_from_p0 | ...], move on. Only a budget smaller than
        one partition (+ its boundary rows) fails.
        """
        buf = self.host_buffers[layer_id]
        deps = self._dependency_set(target_pid, boundaries)
        demand = sum(self._partition_bytes(layer_id, p) for p in deps)
        row_bytes = buf.embedding_dim * buf._element_size
        # The C++ gather index_selects every boundary block into a CPU tensor
        # (keep_alive) that lives until its copies to the GPU finished.
        copy_bytes = row_bytes * sum(
            b.numel()
            for j, b in enumerate(boundaries)
            if j != target_pid and b is not None
        )

        # The previous gather was already waited for; make sure its copies
        # (and keep_alive) are gone before its reservation is dropped.
        self._wait_pool("h2d")
        self._inflight_gather_keys = set()
        self._inflight_gather_bytes = 0

        # Partitions loaded for upload cannot be evicted right now.
        if demand + copy_bytes <= self._activation_cache_budget - self._upload_bytes:
            self.ensure_dependencies_in_host(
                layer_id, target_pid, phase, boundaries=boundaries,
                extra_bytes=copy_bytes,
            )
            buf.async_gather(phase, target_pid, gpu_target, boundaries, stream)
            # The copies run after we return: keep these partitions resident
            # and the boundary copy counted until they are waited for.
            self._inflight_gather_keys = {(layer_id, p) for p in deps}
            self._inflight_gather_bytes = copy_bytes
            return

        segs = self._row_segments(
            buf.part_sizes, target_pid, boundaries, gpu_target.size(0)
        )
        # Use what is cached before anything gets evicted.
        segs.sort(key=lambda s: (layer_id, s[0]) not in self._cached_partitions)

        print(
            f"[cache] layer {layer_id} partition {target_pid}: dependencies + "
            f"boundary copy need {(demand + copy_bytes) / 1024**3:.2f} GB > budget "
            f"{self._activation_cache_budget / 1024**3:.2f} GB, swapping "
            f"{len(segs)} partitions one at a time"
        )

        stream.wait_stream(torch.cuda.current_stream(gpu_target.device))
        for src_pid, idx, row, n in segs:
            temp = 0 if idx is None else n * row_bytes   # index_select result
            self.ensure_dependencies_in_host(
                layer_id, target_pid, phase, dependencies={src_pid},
                extra_bytes=temp,
            )
            src = buf[src_pid]
            self._temp_bytes = temp
            # Blocking copies: the partition may be evicted right after.
            with torch.cuda.stream(stream), torch.no_grad():
                if idx is None:
                    gpu_target[row:row + n].copy_(src)
                else:
                    gpu_target[row:row + n].copy_(src.index_select(0, idx))
            stream.synchronize()
            self._temp_bytes = 0

    def _row_segments(
        self,
        part_sizes: List[int],
        target_pid: int,
        boundaries: List[Optional[torch.Tensor]],
        total_rows: int,
    ) -> List[Tuple[int, Optional[torch.Tensor], int, int]]:
        """(src_pid, row indices or None, first GPU row, num rows) of each source
        partition, in the gather/scatter layout [intra | boundary_from_p0 | ...]."""
        segs = []
        off = part_sizes[target_pid]
        segs.append((target_pid, None, 0, off))
        for j in range(self.num_parts):
            b = boundaries[j]
            if j != target_pid and b is not None and b.numel() > 0:
                segs.append((j, b, off, b.numel()))
                off += b.numel()
        assert off == total_rows, (
            f"Gather layout has {off} rows, GPU tensor has {total_rows}"
        )
        return segs

    # ------------------------------------------------------------------
    # Gradients (io_mode=base, partition_lru): same budget and LRU as the
    # activation partitions.
    # ------------------------------------------------------------------

    @staticmethod
    def _grad_key(layer_id: int, pid: int) -> Tuple[str, int, int]:
        return ("g", layer_id, pid)

    def _grad_bytes(self, layer_id: int, pid: int) -> int:
        return self.grad_buffers[layer_id].partition_nbytes(pid)

    @property
    def resident_bytes(self) -> int:
        """DRAM held by everything the budget covers: cached activation and
        gradient partitions, partitions loaded for upload, and the CPU copies
        made by gathers and scatters."""
        return (
            self._resident_activation_bytes
            + self._resident_gradient_bytes
            + self._upload_bytes
            + self._inflight_gather_bytes
            + self._inflight_scatter_bytes
            + self._temp_bytes
        )

    def ensure_gradients_in_host(
        self, layer_id: int, pids: Set[int], extra_bytes: int = 0
    ) -> bool:
        """Make the gradient partitions `pids` of layer_id resident for
        accumulation, evicting LRU partitions (activations or other gradients)
        as needed, leaving room for `extra_bytes` more. An evicted gradient
        partition comes back from its file, a new one starts at zero. Returns
        False if they cannot all fit."""
        keys = {self._grad_key(layer_id, p) for p in pids}
        missing = []
        for p in sorted(pids):
            key = self._grad_key(layer_id, p)
            if key in self._cached_partitions:
                self._cached_partitions.move_to_end(key)
                self._hits += 1
            else:
                missing.append(p)
                self._misses += 1
        need = sum(self._grad_bytes(layer_id, p) for p in missing)
        if not self._make_room(need + extra_bytes, keys):
            return False
        buf = self.grad_buffers[layer_id]
        for p in missing:
            if (layer_id, p) in self._spilled_gradients:
                buf.storage_to_cpu("backward", pid=p)
                self._spilled_gradients.discard((layer_id, p))
                print(f"[cache] Reloaded gradient partition [Layer = {layer_id} | PID = {p}]")
            else:
                buf.zero_partition(p)
            self._cached_partitions[self._grad_key(layer_id, p)] = True
            self._resident_gradient_bytes += self._grad_bytes(layer_id, p)
            self._cache_tracker.add_gradient(layer_id, p, self._grad_bytes(layer_id, p))
        return True

    def scatter_gradients(
        self,
        layer_id: int,
        pid: int,
        grad: torch.Tensor,
        boundaries: List[Optional[torch.Tensor]],
        stream: torch.cuda.Stream,
    ) -> None:
        """Base gradient offload: scatter-add partition pid's input gradient
        (GPU, layout [intra | boundary_from_p0 | ...]) into the DRAM gradient
        partitions of layer_id, within the cache budget.

        All destination partitions + the scatter's CPU copies fit: make them
        resident, one async C++ scatter (original). Otherwise: one destination
        partition at a time (cached ones first), each made resident (evicting
        LRU partitions) and added to synchronously.
        """
        buf = self.grad_buffers[layer_id]
        deps = self._dependency_set(pid, boundaries)
        row_bytes = buf.embedding_dim * buf._element_size
        # The C++ scatter clones the target partition (prev_values) and copies
        # one boundary block at a time into a temporary CPU tensor.
        largest_block = max(
            (b.numel() for j, b in enumerate(boundaries) if j != pid and b is not None),
            default=0,
        )
        copy_bytes = self._grad_bytes(layer_id, pid) + largest_block * row_bytes

        # The previous scatter was already waited for; make sure it is done
        # before its reservation is dropped.
        self._wait_pool("d2h")
        self._inflight_scatter_bytes = 0

        if self.ensure_gradients_in_host(layer_id, deps, extra_bytes=copy_bytes):
            buf.async_scatter(pid, grad, boundaries, stream)
            self._inflight_scatter_bytes = copy_bytes
            return

        segs = self._row_segments(buf.part_sizes, pid, boundaries, grad.size(0))
        segs.sort(
            key=lambda s: self._grad_key(layer_id, s[0]) not in self._cached_partitions
        )
        demand = sum(self._grad_bytes(layer_id, p) for p in deps)
        print(
            f"[cache] layer {layer_id} partition {pid}: gradients + copies need "
            f"{(demand + copy_bytes) / 1024**3:.2f} GB > budget "
            f"{self._activation_cache_budget / 1024**3:.2f} GB, accumulating "
            f"{len(segs)} partitions one at a time"
        )
        for src_pid, idx, row, n in segs:
            temp = n * row_bytes   # rows copied to the CPU
            if not self.ensure_gradients_in_host(layer_id, {src_pid}, extra_bytes=temp):
                raise MemoryError(
                    f"Gradient partition [Layer = {layer_id} | PID = {src_pid}] "
                    f"+ its {temp} byte block do not fit the "
                    f"{self._activation_cache_budget} byte cache budget"
                )
            self._temp_bytes = temp
            with torch.cuda.stream(stream), torch.no_grad():
                rows = grad[row:row + n].cpu()   # blocking
            dst = buf[src_pid]
            if idx is None:
                dst.add_(rows)
            else:
                dst.index_add_(0, idx, rows)
            del rows
            self._temp_bytes = 0

    # ------------------------------------------------------------------
    # Partitions loaded only to be uploaded to the GPU: counted in the
    # budget, never evicted (the copy may still be running).
    # ------------------------------------------------------------------

    def _load_for_upload(
        self, kind: str, buf: HostBuffer, layer_id: int, pid: int, phase: str,
        zero_if_missing: bool,
    ) -> int:
        nbytes = buf.partition_nbytes(pid)
        if not self._make_room(nbytes, set()):
            raise MemoryError(
                f"Partition [{kind} | Layer = {layer_id} | PID = {pid}] "
                f"({nbytes} bytes) does not fit the "
                f"{self._activation_cache_budget} byte cache budget"
            )
        if zero_if_missing and not buf.storage_exists(pid):
            buf.zero_partition(pid)
        else:
            buf.storage_to_cpu(phase, pid=pid)
        self._uploads[(kind, layer_id, pid)] = nbytes
        self._upload_bytes += nbytes
        return nbytes

    def _release_upload(self, kind: str, buf: HostBuffer, layer_id: int, pid: int) -> bool:
        buf.release(pid)
        nbytes = self._uploads.pop((kind, layer_id, pid), None)
        if nbytes is None:
            return False
        self._upload_bytes = max(0, self._upload_bytes - nbytes)
        return True

    def load_gradient_for_upload(self, layer_id: int, pid: int) -> None:
        """Load a finished gradient partition (file, else zero) into DRAM so it
        can be uploaded to the GPU. Free it with release_gradient_upload."""
        print(f"[Layer = {layer_id} | PID = {pid}] Gradients move SSD --> CPU")
        nbytes = self._load_for_upload(
            "g", self.grad_buffers[layer_id], layer_id, pid, "gradient",
            zero_if_missing=True,
        )
        self._cache_tracker.add_gradient(layer_id, pid, nbytes)

    def release_gradient_upload(self, layer_id: int, pid: int) -> None:
        """Free a gradient partition loaded by load_gradient_for_upload."""
        if self._release_upload("g", self.grad_buffers[layer_id], layer_id, pid):
            self._cache_tracker.remove_gradient(layer_id, pid)

    def load_activation_for_upload(self, layer_id: int, pid: int, phase: str) -> None:
        """Load an activation partition from its file into DRAM so it can be
        uploaded to the GPU (e.g. the last layer for the loss). Free it with
        release_activation_upload."""
        self._load_for_upload(
            "a", self.host_buffers[layer_id], layer_id, pid, phase,
            zero_if_missing=False,
        )

    def release_activation_upload(self, layer_id: int, pid: int) -> None:
        """Free an activation partition loaded by load_activation_for_upload."""
        self._release_upload("a", self.host_buffers[layer_id], layer_id, pid)

    def _evict_gradient(self, layer_id: int, pid: int) -> None:
        """Spill a partially accumulated gradient partition to its file."""
        # Scatters into it may still be running on the D2H streams.
        self._sync_transfers()
        buf = self.grad_buffers[layer_id]
        buf.cpu_to_storage(pid)
        buf.release(pid)
        self._spilled_gradients.add((layer_id, pid))
        self._cached_partitions.pop(self._grad_key(layer_id, pid), None)
        self._resident_gradient_bytes = max(
            0, self._resident_gradient_bytes - self._grad_bytes(layer_id, pid)
        )
        self._cache_tracker.remove_gradient(layer_id, pid)
        print(f"GRADIENT EVICTED [Layer = {layer_id} | PID = {pid}]")

    def _drop_gradient_entry(self, layer_id: int, pid: int) -> None:
        """Forget a resident gradient partition (caller freed or flushed it)."""
        if self._cached_partitions.pop(self._grad_key(layer_id, pid), None) is not None:
            self._resident_gradient_bytes = max(
                0, self._resident_gradient_bytes - self._grad_bytes(layer_id, pid)
            )

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def _make_room(self, need: int, protected: Set[Tuple]) -> bool:
        """Evict LRU partitions (activations or gradients, not `protected`)
        until `need` more bytes fit the budget. When nothing else can go,
        wait for the running gather/scatter so its partitions can be evicted
        and its CPU copies are freed."""
        while self.resident_bytes + need > self._activation_cache_budget:
            key = self._pop_evictable_partition(protected | self._inflight_gather_keys)
            if key is None:
                if not (
                    self._inflight_gather_keys
                    or self._inflight_gather_bytes
                    or self._inflight_scatter_bytes
                ):
                    return False
                self._sync_transfers()
                continue
            self._evict_key(key)
        return True

    def _evict_key(self, key: Tuple) -> None:
        if len(key) == 3:
            self._evict_gradient(key[1], key[2])
        else:
            self._evict_partition(key[0], key[1])

    def _copy_ops(self):
        return getattr(self.host_buffers[0], "_ops", None) if self.host_buffers else None

    def _wait_pool(self, which: str) -> None:
        """Wait for the C++ H2D or D2H copy thread to finish its tasks (each
        gather/scatter task syncs its stream before it ends)."""
        ops = self._copy_ops()
        if ops is None:
            return
        try:
            (ops.h2d_synchronize if which == "h2d" else ops.d2h_synchronize)()
        except RuntimeError:
            pass  # nothing pending

    def _sync_transfers(self) -> None:
        """Wait for every pending DRAM <-> GPU copy (gathers, scatters, uploads)."""
        self._wait_pool("d2h")
        self._wait_pool("h2d")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._inflight_gather_keys = set()
        self._inflight_gather_bytes = 0
        self._inflight_scatter_bytes = 0

    def on_layer_complete(self, layer_id: int, pids: int) -> None:
        """Called after forward layer completes. Decide what to flush."""
        output_layer = layer_id + 1
        if output_layer >= len(self.host_buffers):
            return

        if self._mode == "lru_layer":
            if output_layer in self._cached_layers:
                self._cached_layers.move_to_end(output_layer)
                return

            layer_bytes = self._layer_bytes(output_layer)
            if layer_bytes > self._activation_cache_budget:
                raise MemoryError(
                    f"Layer {output_layer} requires {layer_bytes} bytes, but "
                    f"only {self._activation_cache_budget} activation-cache "
                    "bytes remain"
                )
            while (
                self._resident_activation_bytes + layer_bytes
                > self._activation_cache_budget
            ):
                if not self._cached_layers:
                    raise MemoryError(
                        f"Cannot keep layer {output_layer} within "
                        f"{self._activation_cache_budget} bytes"
                    )
                evict_id, _ = self._cached_layers.popitem(last=False)
                self._evict_layer(evict_id)

            self._cached_layers[output_layer] = True
            self._resident_activation_bytes += layer_bytes
            return

        # Partition-wise mode still bypasses outputs to storage and loads only
        # demanded partitions on the next gather/regather.

        print("LAYER COMPLETE")

    def on_backward_layer_complete(self, layer_id: int) -> None:
        """Called after backward layer completes. Flush gradients to storage."""
        if self.grad_buffers[layer_id] is not None:
            if self._mode == "partition_lru":
                # The trainer waited for all scatters of this layer.
                self._wait_pool("d2h")
                self._inflight_scatter_bytes = 0
                # Resident partitions go to their files now; spilled ones are
                # already there.
                for pid in self.grad_buffers[layer_id].allocated_pids():
                    self.grad_buffers[layer_id].cpu_to_storage(pid)
                    self.grad_buffers[layer_id].release(pid)

                    print(f"Flushing gradients to SSD [Layer = {layer_id} | Partition = {pid}]\n")
                    self._cache_tracker.remove_gradient(layer_id, pid)
                    self._drop_gradient_entry(layer_id, pid)
                self._spilled_gradients = {
                    k for k in self._spilled_gradients if k[0] != layer_id
                }
                return
            self.grad_buffers[layer_id].cpu_to_storage()

    def prepare_layer(self, phase, layer_id: int) -> None:
        """Alias for ensure_in_host (used in backward for regathering)."""
        self.ensure_in_host(layer_id)

    def _evict_layer(self, layer_id: int) -> None:
        """Evict a layer's activations from host cache.

        Ensure a storage copy exists before freeing host memory. For true
        layer-wise LRU, newly produced layers stay resident until budget
        pressure actually evicts them.
        """
        buf = self.host_buffers[layer_id]
        for pid in range(self.num_parts):
            buf.ensure_storage_copy(pid)
            buf.release(pid)
            print(f"Evict partition {pid} from CPU cache")
        self._resident_activation_bytes = max(
            0, self._resident_activation_bytes - self._layer_bytes(layer_id)
        )

        print("LAYER EVICTED")
        self.cached_partitions()
        
    def _evict_partition(self, layer_id: int, pid: int) -> None:
        """Evict a single partition's activations from host cache.

        Ensure a storage copy exists before freeing host memory.
        """
        self._cache_tracker.remove_partition(layer_id, pid)
        key = (layer_id, pid)
        buf = self.host_buffers[layer_id]
        buf.ensure_storage_copy(pid)
        buf.release(pid)
        self._resident_activation_bytes = max(
            0, self._resident_activation_bytes - self._partition_bytes(layer_id, pid)
        )
        self._cached_partitions.pop(key, None)

        print(f"PARTITION EVICTED [Layer = {layer_id} | PID = {pid}]")

    def _layer_bytes(self, layer_id: int) -> int:
        """Estimate bytes for one layer's activations."""
        return self.host_buffers[layer_id].total_nbytes

    def _partition_bytes(self, layer_id: int, pid: int) -> int:
        return self.host_buffers[layer_id].partition_nbytes(pid)

    def _dependency_set(
        self,
        target_pid: int,
        boundaries: Optional[List[Optional[torch.Tensor]]] = None,
        dependencies: Optional[Set[int]] = None,
    ) -> Set[int]:
        if dependencies is not None:
            return set(dependencies)
        if boundaries is not None:
            deps = {target_pid}
            for src_pid, boundary in enumerate(boundaries):
                if (
                    src_pid != target_pid
                    and boundary is not None
                    and boundary.numel() > 0
                ):
                    deps.add(src_pid)
            return deps
        if self.dependency_sets is not None:
            return set(self.dependency_sets[target_pid])
        return {target_pid}

    def _pop_evictable_partition(self, protected: Set[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        for key in list(self._cached_partitions.keys()):
            if key not in protected:
                self._cached_partitions.pop(key)
                return key
        return None
    
    def _build_cache_tracker(self, parts) -> PartitionCacheTracker:

        partitions=[[],[],[]]
        gradients=[[],[],[]]


        for i in range(parts):
            partitions[0].append(0)
            partitions[1].append(0)
            partitions[2].append(0)

            gradients[0].append(0)
            gradients[1].append(0)
            gradients[2].append(0)

        return PartitionCacheTracker(
            partitions,
            gradients,
            0,
            0
        )
    
    def reset(self) -> None:
        """Reset cache state for new epoch."""
        if self._mode == "lru_layer":
            for layer_id in list(self._cached_layers.keys()):
                self._evict_layer(layer_id)
        elif self._mode == "partition_lru":
            self._sync_transfers()
            for key in list(self._cached_partitions.keys()):
                if len(key) == 3:
                    # Leftover gradient: stale for the new epoch, just free it.
                    self.grad_buffers[key[1]].release(key[2])
                    self._cache_tracker.remove_gradient(key[1], key[2])
                else:
                    self._evict_partition(key[0], key[1])
            for kind, layer_id, pid in self._uploads:
                if kind == "g":
                    self.grad_buffers[layer_id].release(pid)
                    self._cache_tracker.remove_gradient(layer_id, pid)
                else:
                    self.host_buffers[layer_id].release(pid)
        self._cached_layers.clear()
        self._cached_partitions.clear()
        self._hits = 0
        self._misses = 0
        self._resident_activation_bytes = 0
        self._resident_gradient_bytes = 0
        self._spilled_gradients.clear()
        self._uploads.clear()
        self._upload_bytes = 0
        self._inflight_gather_keys = set()
        self._inflight_gather_bytes = 0
        self._inflight_scatter_bytes = 0
        self._temp_bytes = 0

    def add_gradient_relay(self, layer, pid):
        self._cache_tracker.add_gradient(layer,pid,self._partition_bytes(layer,pid))

    def cache_tracker_print(self):
        self._cache_tracker.print_cache()
