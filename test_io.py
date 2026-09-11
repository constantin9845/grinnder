import os
import ctypes
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor

libc = ctypes.CDLL(None)
ALIGNMENT = 4096
NUM_WORKERS = 128  # Pushes Queue Depth to 128 to better saturate NVMe channels

file1 = "/mnt/nvme/feat_l0_p7.pt"
file2 = "/mnt/nvme/feat_l0_p8.pt"
file3 = "/mnt/nvme/feat_l0_p9.pt"

flags = os.O_RDONLY | os.O_DIRECT

# Setup thread-local storage so worker threads allocate aligned memory once
thread_local = threading.local()

def get_thread_buffer(buf_size):
    """Allocates a persistent aligned buffer per thread to eliminate heap allocation overhead in loops."""
    if not hasattr(thread_local, "buf_ptr"):
        thread_local.buf_ptr = ctypes.c_void_p()
        if libc.posix_memalign(ctypes.byref(thread_local.buf_ptr), ALIGNMENT, buf_size) != 0:
            raise OSError("posix_memalign failed in thread setup")
        thread_local.memview = memoryview((ctypes.c_char * buf_size).from_address(thread_local.buf_ptr.value))
    return thread_local.memview


# =====================================================================
# 1. FULL FILE SINGLE-BUFFER READ (64 MB Chunks)
# =====================================================================
print("--- Test 1: Full File Read (64 MB Sequential Chunks) ---")
FILENAME = file1
CHUNK_SIZE = 64 * 1024 * 1024  # 64 MB

fd = os.open(FILENAME, flags)

try:
    file_size = os.lseek(fd, 0, os.SEEK_END)
    os.lseek(fd, 0, os.SEEK_SET)

    buf_ptr = ctypes.c_void_p()
    if libc.posix_memalign(ctypes.byref(buf_ptr), ALIGNMENT, CHUNK_SIZE) != 0:
        raise OSError("Memory allocation failed")

    memview = memoryview((ctypes.c_char * CHUNK_SIZE).from_address(buf_ptr.value))

    offset = 0
    total_bytes_read = 0

    t0 = time.perf_counter_ns()

    while offset < file_size:
        bytes_read = os.preadv(fd, [memview], offset)
        if bytes_read == 0:
            break

        offset += bytes_read
        total_bytes_read += bytes_read

    tn = time.perf_counter_ns()
    libc.free(buf_ptr)

    elapsed_sec = (tn - t0) / 1e9
    throughput_mb = (total_bytes_read / (1024 * 1024)) / elapsed_sec if elapsed_sec > 0 else 0

    print(f"Finished reading {total_bytes_read:,} / {file_size:,} bytes.")
    print(f"Time Taken  : {elapsed_sec:.6f} seconds")
    print(f"Throughput  : {throughput_mb:.2f} MB/s\n")

finally:
    os.close(fd)


# =====================================================================
# 2. SEQUENTIAL PARTIAL READS (Parallel 4 KB Chunks)
# =====================================================================
print(f"--- Test 2: Sequential 4 KB Reads ({NUM_WORKERS} Threads / QD={NUM_WORKERS}) ---")
FILENAME = file2
CHUNK_SIZE = 4096
fd = os.open(FILENAME, flags)

try:
    file_size = os.lseek(fd, 0, os.SEEK_END)
    os.lseek(fd, 0, os.SEEK_SET)

    sequential_offsets = list(range(0, file_size, CHUNK_SIZE))

    def read_block_4k(offset):
        memview = get_thread_buffer(CHUNK_SIZE)
        return os.preadv(fd, [memview], offset)

    t0 = time.perf_counter_ns()

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(executor.map(read_block_4k, sequential_offsets))

    tn = time.perf_counter_ns()

    total_bytes_read = sum(results)
    elapsed_sec = (tn - t0) / 1e9
    throughput_mb = (total_bytes_read / (1024 * 1024)) / elapsed_sec if elapsed_sec > 0 else 0
    iops = len(sequential_offsets) / elapsed_sec if elapsed_sec > 0 else 0

    print(f"Finished reading {total_bytes_read:,} / {file_size:,} bytes in 4KB chunks.")
    print(f"Time Taken  : {elapsed_sec:.6f} seconds")
    print(f"Throughput  : {throughput_mb:.2f} MB/s")
    print(f"Read IOPS   : {iops:.2f} IOPS\n")

finally:
    os.close(fd)


# =====================================================================
# 3. SCATTERED RANDOM PARTIAL READS (Parallel 13,700 Offsets)
# =====================================================================
print(f"--- Test 3: Scattered 4 KB Reads ({NUM_WORKERS} Threads / QD={NUM_WORKERS}) ---")
FILENAME = file3
NUM_READS = 13700
CHUNK_SIZE = 4096

fd = os.open(FILENAME, flags)

try:
    file_size = os.lseek(fd, 0, os.SEEK_END)
    os.lseek(fd, 0, os.SEEK_SET)

    total_blocks = file_size // CHUNK_SIZE
    if total_blocks == 0:
        raise ValueError("File is smaller than 4096 bytes.")

    scattered_offsets = [random.randrange(0, total_blocks) * CHUNK_SIZE for _ in range(NUM_READS)]

    def read_random_block_4k(offset):
        memview = get_thread_buffer(CHUNK_SIZE)
        return os.preadv(fd, [memview], offset)

    t0 = time.perf_counter_ns()

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(executor.map(read_random_block_4k, scattered_offsets))

    tn = time.perf_counter_ns()

    total_bytes_read = sum(results)
    elapsed_sec = (tn - t0) / 1e9
    throughput_mb = (total_bytes_read / (1024 * 1024)) / elapsed_sec if elapsed_sec > 0 else 0
    iops = len(scattered_offsets) / elapsed_sec if elapsed_sec > 0 else 0

    print(f"File Size   : {file_size / (1024**3):.2f} GB ({total_blocks:,} total blocks)")
    print(f"Completed   : {len(scattered_offsets):,} scattered reads ({total_bytes_read / (1024**2):.2f} MB total data)")
    print(f"Time Taken  : {elapsed_sec:.4f} seconds")
    print(f"Throughput  : {throughput_mb:.2f} MB/s")
    print(f"Random IOPS : {iops:.2f} IOPS\n")

finally:
    os.close(fd)


# =====================================================================
# 4. OPTIONAL DEMO: SEQUENTIAL 128 KB READS (Parallel)
# =====================================================================
print(f"--- Test 4: Sequential 128 KB Reads ({NUM_WORKERS} Threads / QD={NUM_WORKERS}) ---")
FILENAME = file2
CHUNK_SIZE_128K = 128 * 1024
fd = os.open(FILENAME, flags)

try:
    file_size = os.lseek(fd, 0, os.SEEK_END)
    os.lseek(fd, 0, os.SEEK_SET)

    offsets_128k = list(range(0, file_size, CHUNK_SIZE_128K))

    # Clean thread-local buffer state for 128KB
    thread_local = threading.local()

    def read_block_128k(offset):
        memview = get_thread_buffer(CHUNK_SIZE_128K)
        return os.preadv(fd, [memview], offset)

    t0 = time.perf_counter_ns()

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(executor.map(read_block_128k, offsets_128k))

    tn = time.perf_counter_ns()

    total_bytes_read = sum(results)
    elapsed_sec = (tn - t0) / 1e9
    throughput_mb = (total_bytes_read / (1024 * 1024)) / elapsed_sec if elapsed_sec > 0 else 0

    print(f"Finished reading {total_bytes_read:,} / {file_size:,} bytes in 128KB chunks.")
    print(f"Time Taken  : {elapsed_sec:.6f} seconds")
    print(f"Throughput  : {throughput_mb:.2f} MB/s\n")

finally:
    os.close(fd)