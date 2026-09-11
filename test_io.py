import os
import ctypes
import time
import random

FILENAME = "/mnt/nvme/feat_l0_p0.pt"
ALIGNMENT = 4096
CHUNK_SIZE = 4096

libc = ctypes.CDLL(None)


# =====================================================================
# 1. FULL FILE SINGLE-BUFFER READ
# =====================================================================
print("--- Test 1: Full File Read ---")
flags = os.O_RDONLY | os.O_DIRECT
fd = os.open(FILENAME, flags)

try:
    file_size = os.lseek(fd, 0, os.SEEK_END)
    os.lseek(fd, 0, os.SEEK_SET)

    # Allocate aligned buffer for the full file (rounded up to nearest 4096)
    alloc_size = ((file_size + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT

    buf_ptr = ctypes.c_void_p()
    if libc.posix_memalign(ctypes.byref(buf_ptr), ALIGNMENT, alloc_size) != 0:
        raise OSError("posix_memalign failed")

    aligned_memview = memoryview((ctypes.c_char * alloc_size).from_address(buf_ptr.value))

    t0 = time.perf_counter_ns()
    
    # Low-level Direct I/O read into aligned memory
    bytes_read = os.preadv(fd, [aligned_memview], 0)

    tn = time.perf_counter_ns()

    elapsed_sec = (tn - t0) / 1e9
    throughput_mb = (bytes_read / (1024 * 1024)) / elapsed_sec if elapsed_sec > 0 else 0

    print(f"Successfully read {bytes_read:,} / {file_size:,} bytes using O_DIRECT.")
    print(f"Time Taken  : {elapsed_sec:.6f} seconds")
    print(f"Throughput  : {throughput_mb:.2f} MB/s\n")

    libc.free(buf_ptr)

finally:
    os.close(fd)


# =====================================================================
# 2. SEQUENTIAL PARTIAL READS (Full File in 4 KB Chunks)
# =====================================================================
print("--- Test 2: Sequential 4 KB Reads ---")
fd = os.open(FILENAME, flags)

try:
    file_size = os.lseek(fd, 0, os.SEEK_END)
    os.lseek(fd, 0, os.SEEK_SET)

    buf_ptr = ctypes.c_void_p()
    if libc.posix_memalign(ctypes.byref(buf_ptr), ALIGNMENT, CHUNK_SIZE) != 0:
        raise OSError("posix_memalign failed")

    aligned_memview = memoryview((ctypes.c_char * CHUNK_SIZE).from_address(buf_ptr.value))

    offset = 0
    total_bytes_read = 0

    t0 = time.perf_counter_ns()

    while offset < file_size:
        bytes_read = os.preadv(fd, [aligned_memview], offset)
        if bytes_read == 0:
            break  # End of file reached

        offset += bytes_read
        total_bytes_read += bytes_read

    tn = time.perf_counter_ns()

    libc.free(buf_ptr)

    elapsed_sec = (tn - t0) / 1e9
    throughput_mb = (total_bytes_read / (1024 * 1024)) / elapsed_sec if elapsed_sec > 0 else 0

    print(f"Finished reading {total_bytes_read:,} / {file_size:,} bytes in 4KB chunks.")
    print(f"Time Taken  : {elapsed_sec:.6f} seconds")
    print(f"Throughput  : {throughput_mb:.2f} MB/s\n")

finally:
    os.close(fd)


# =====================================================================
# 3. SCATTERED RANDOM PARTIAL READS (13,700 Offsets)
# =====================================================================
print("--- Test 3: Scattered 4 KB Reads ---")
NUM_READS = 13700

fd = os.open(FILENAME, flags)

try:
    file_size = os.lseek(fd, 0, os.SEEK_END)
    os.lseek(fd, 0, os.SEEK_SET)

    total_blocks = file_size // CHUNK_SIZE
    if total_blocks == 0:
        raise ValueError("File is smaller than 4096 bytes.")

    # Generate 13,700 valid block offsets
    mapping = [random.randrange(0, total_blocks) * CHUNK_SIZE for _ in range(NUM_READS)]

    buf_ptr = ctypes.c_void_p()
    if libc.posix_memalign(ctypes.byref(buf_ptr), ALIGNMENT, CHUNK_SIZE) != 0:
        raise OSError("posix_memalign failed")

    aligned_memview = memoryview((ctypes.c_char * CHUNK_SIZE).from_address(buf_ptr.value))

    total_bytes_read = 0

    t0 = time.perf_counter_ns()

    for offset in mapping:
        bytes_read = os.preadv(fd, [aligned_memview], offset)
        if bytes_read == 0:
            continue

        total_bytes_read += bytes_read

    tn = time.perf_counter_ns()

    libc.free(buf_ptr)

    elapsed_sec = (tn - t0) / 1e9
    throughput_mb = (total_bytes_read / (1024 * 1024)) / elapsed_sec if elapsed_sec > 0 else 0
    iops = len(mapping) / elapsed_sec if elapsed_sec > 0 else 0

    print(f"File Size   : {file_size / (1024**3):.2f} GB ({total_blocks:,} total blocks)")
    print(f"Completed   : {len(mapping):,} scattered reads ({total_bytes_read / (1024**2):.2f} MB total data)")
    print(f"Time Taken  : {elapsed_sec:.4f} seconds")
    print(f"Throughput  : {throughput_mb:.2f} MB/s")
    print(f"Random IOPS : {iops:.2f} IOPS")

finally:
    os.close(fd)