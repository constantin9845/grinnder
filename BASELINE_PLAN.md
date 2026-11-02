# Plan: baseline (`base`) vs proposal (`ours`) I/O modes

Status: agreed, not implemented yet.

## Arguments (`examples/train_igb_metis.py`)

- `--io_mode base|ours` (also `--io-mode`), default `ours` -> `GriNNderConfig.io_mode`.
  The existing `--mode hongtu|grinnder` stays.
- `--cache-size 16GB` (also `--cache_size`; accepts `16`, `16GB`, `512MB`) -> `GriNNderConfig.cache_size_gb`.
  Base only. Used as the DRAM budget for cached activation partitions as given
  (no safety margin / graph metadata subtracted). Omitted: budget computed as today.
- Base forces `cache_mode = partition_lru` (original per-partition LRU cache).

## `ours` (proposal) -- already implemented, unchanged

Everything SSD <-> GPU through the host FTL on the raw device; DRAM only as staging inside the C++ functions.

| Phase | Path |
|---|---|
| Forward gather / backward re-gather | target + boundary rows only, FTL row offsets (`gather_partitions_nvme_async`) |
| Forward output | GPU -> SSD, FTL addresses (`gpu_write_nvme_async`) |
| Loss | whole last-layer partition SSD -> GPU (FTL) |
| Gradients | GPU read-add-write of touched rows on SSD (`accumulate_raw`); upload reads only written rows |

## `base` (original GriNNder, no FTL)

Restore the original code path from commit `54423c4`, gated by `io_mode`. Files on the ext4 mount
(`/mnt/nvme/feat_l*_p*.pt`, `grad_l*_p*.pt`).

| Phase | Path |
|---|---|
| Forward gather / backward re-gather | `ensure_dependencies_in_host` loads **whole** dependency partitions file -> DRAM (io_uring), LRU eviction; then **C++ `gather_partitions`** DRAM -> GPU (restore it; remove the `2 == 3` Python fallback in `HostBuffer.async_gather`) |
| Forward output | GPU -> file via kvikio `CuFile` write (original) |
| Loss | `storage_to_cpu` (whole partition, io_uring) -> `async_upload` |
| Gradients | `async_scatter` into DRAM gradient partitions (C++); upload: load file if it exists else zero, `async_upload`; layer end: `on_backward_layer_complete` flushes to files (io_uring) |
| Prefill (layer 0) | `cpu_to_storage` -> file (io_uring) |

### New: swapping when one partition's dependencies exceed `--cache-size`

- Fits: as original (load all deps, one C++ gather).
- Does not fit: instead of `MemoryError` (`partition_cache.py` `ensure_dependencies_in_host`), work in chunks:
  load as many dependency partitions as fit (target first), copy their rows into their fixed GPU row
  ranges (`[intra | boundary_from_p0 | ...]`), wait for the copy, evict, load the next chunk.
- Hard limit only if `--cache-size` < one partition.
- Print `PartitionCache` hits / misses / evictions per layer.

### Gradients in base

Original behavior: whole gradient layer stays in DRAM during the backward layer (outside the cache
budget), flushed at layer end. Print a warning if activations + gradients exceed `--cache-size`.

## Code structure

- `StorageBackend`: create `FTL()` only for `ours`. Base: `host_read`/`host_write` = original io_uring
  file I/O, `gpu_write` = original kvikio file write, `exists` = file check. Ours: FTL as now.
- `HostBuffer`: `storage_exists` = file (base) / FTL table (ours). `cpu_to_storage` waits for its
  io_uring write again (original).
- Trainer helpers branching on `io_mode`: `_gather_features`, `_load_final_activation`,
  `_prefetch_gradient`, `_begin_gradient_layer`, `_end_gradient_layer` (replace duplicated call sites).
- `GradOffload`: `accumulate_raw` (ours) or `async_scatter` (base).
- `ScatteredCheckpoint`: per-partition wait (already in place) works for both.
- No C++ changes; no rebuild needed.

## Running

- ours: raw device, not mounted.
- base: ext4 formatted + mounted at `/mnt/nvme`, `--io_mode base --cache-size 16GB`,
  `KVIKIO_COMPAT_MODE=ON` (kvikio uses pread/pwrite instead of cuFile, which hung under FEMU).
- User formats FEMU per mode; code assumes its own storage layout is valid.
