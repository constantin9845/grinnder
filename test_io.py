import os
import ctypes
import time
import random
from concurrent.futures import ThreadPoolExecutor

libc = ctypes.CDLL(None)
ALIGNMENT = 4096
NUM_WORKERS = 32  # Queue depth sent to NVMe controller

file1 = "/mnt/nvme/feat_l0_p7.pt"
file2 = "/mnt/nvme/feat_l0_p8.pt"
file3 = "/mnt/nvme/feat_l0_p9.pt"


# =====================================================================
# 1. FULL FILE SINGLE-BUFFER READ (64 MB Chunks)
# =====================================================================
print("--- Test 1: Full File Read ---")
FILENAME = file1
CHUNK_SIZE = 64 * 1024 * 1024  # 64 MB
flags = os.O_RDONLY | os.O_DIRECT

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
print(f"--- Test 2: Sequential 4 KB Reads ({NUM_WORKERS} Threads) ---")
FILENAME = file2
CHUNK_SIZE = 4096
fd = os.open(FILENAME, flags)

try:
    file_size = os.lseek(fd, 0, os.SEEK_END)
    os.lseek(fd, 0, os.SEEK_SET)

    # Pre-generate all sequential block offsets
    sequential_offsets = list(range(0, file_size, CHUNK_SIZE))

    # Worker function: Thread-local aligned buffer allocation
    def read_block(offset):
        buf_ptr = ctypes.c_void_p()
        if libc.posix_memalign(ctypes.byref(buf_ptr), ALIGNMENT, CHUNK_SIZE) != 0:
            return 0
        
        memview = memoryview((ctypes.c_char * CHUNK_SIZE).from_address(buf_ptr.value))
        bytes_read = os.preadv(fd, [memview], offset)
        libc.free(buf_ptr)
        return bytes_read

    t0 = time.perf_counter_ns()

    # Issue reads concurrently across worker threads
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(executor.map(read_block, sequential_offsets))

    tn = time.perf_counter_ns()

    total_bytes_read = sum(results)
    elapsed_sec = (tn - t0) / 1e9
    throughput_mb = (total_bytes_read / (1024 * 1024)) / elapsed_sec if elapsed_sec > 0 else 0

    print(f"Finished reading {total_bytes_read:,} / {file_size:,} bytes in 4KB chunks.")
    print(f"Time Taken  : {elapsed_sec:.6f} seconds")
    print(f"Throughput  : {throughput_mb:.2f} MB/s\n")

finally:
    os.close(fd)


# =====================================================================
# 3. SCATTERED RANDOM PARTIAL READS (Parallel 13,700 Offsets)
# =====================================================================
print(f"--- Test 3: Scattered 4 KB Reads ({NUM_WORKERS} Threads) ---")
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

    # Generate 13,700 random offsets
    scattered_offsets = [random.randrange(0, total_blocks) * CHUNK_SIZE for _ in range(NUM_READS)]

    # Worker function: Thread-local aligned buffer allocation
    def read_random_block(offset):
        buf_ptr = ctypes.c_void_p()
        if libc.posix_memalign(ctypes.byref(buf_ptr), ALIGNMENT, CHUNK_SIZE) != 0:
            return 0

        memview = memoryview((ctypes.c_char * CHUNK_SIZE).from_address(buf_ptr.value))
        bytes_read = os.preadv(fd, [memview], offset)
        libc.free(buf_ptr)
        return bytes_read

    t0 = time.perf_counter_ns()

    # Issue scattered reads concurrently across worker threads
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(executor.map(read_random_block, scattered_offsets))

    tn = time.perf_counter_ns()

    total_bytes_read = sum(results)
    elapsed_sec = (tn - t0) / 1e9
    throughput_mb = (total_bytes_read / (1024 * 1024)) / elapsed_sec if elapsed_sec > 0 else 0
    iops = len(scattered_offsets) / elapsed_sec if elapsed_sec > 0 else 0

    print(f"File Size   : {file_size / (1024**3):.2f} GB ({total_blocks:,} total blocks)")
    print(f"Completed   : {len(scattered_offsets):,} scattered reads ({total_bytes_read / (1024**2):.2f} MB total data)")
    print(f"Time Taken  : {elapsed_sec:.4f} seconds")
    print(f"Throughput  : {throughput_mb:.2f} MB/s")
    print(f"Random IOPS : {iops:.2f} IOPS")

finally:
    os.close(fd)