import os
import ctypes
import time
import random
import threading
import fcntl
import struct
from concurrent.futures import ThreadPoolExecutor

libc = ctypes.CDLL(None)

# Device Configuration
BLOCK_DEVICE = "/dev/nvme0n1"  # Target raw block device
SECTOR_SIZE = 512              # Logical sector size (512 or 4096 bytes)
ALIGNMENT = 4096               # Buffer alignment requirement for O_DIRECT
NUM_WORKERS = 128              # Queue Depth / Thread count
TARGET_BYTES_5GB = 5 * 1024 * 1024 * 1024  # 5 GiB per test

# Linux ioctl to get block device size in bytes
BLKGETSIZE64 = 0x80081272

flags = os.O_RDONLY | os.O_DIRECT

thread_local = threading.local()

def get_thread_buffer(buf_size):
    """Allocates a persistent aligned buffer per thread to eliminate heap allocation overhead."""
    if not hasattr(thread_local, "buf_ptr") or getattr(thread_local, "buf_size", 0) < buf_size:
        if hasattr(thread_local, "buf_ptr"):
            libc.free(thread_local.buf_ptr)
        
        thread_local.buf_ptr = ctypes.c_void_p()
        if libc.posix_memalign(ctypes.byref(thread_local.buf_ptr), ALIGNMENT, buf_size) != 0:
            raise OSError("posix_memalign failed in thread setup")
        
        thread_local.memview = memoryview((ctypes.c_char * buf_size).from_address(thread_local.buf_ptr.value))
        thread_local.buf_size = buf_size
        
    return thread_local.memview

def get_device_capacity_lbas(fd, sector_size):
    """Fetches total device capacity in bytes via ioctl and converts to LBAs."""
    buf = struct.pack('L', 0)
    res = fcntl.ioctl(fd, BLKGETSIZE64, buf)
    total_bytes = struct.unpack('L', res)[0]
    total_lbas = total_bytes // sector_size
    return total_bytes, total_lbas


# =====================================================================
# 1. SEQUENTIAL READ - 5 GB (64 MB Chunks via LBAs)
# =====================================================================
print("--- Test 1: Sequential 5 GB Read (64 MB Chunks) ---")
CHUNK_SIZE_BYTES = 64 * 1024 * 1024  # 64 MB

fd = os.open(BLOCK_DEVICE, flags)

try:
    total_bytes, total_lbas = get_device_capacity_lbas(fd, SECTOR_SIZE)
    test_limit_bytes = min(total_bytes, TARGET_BYTES_5GB)
    print(f"Target Read Capacity: {test_limit_bytes / (1024**3):.2f} GB")

    buf_ptr = ctypes.c_void_p()
    if libc.posix_memalign(ctypes.byref(buf_ptr), ALIGNMENT, CHUNK_SIZE_BYTES) != 0:
        raise OSError("Memory allocation failed")

    memview = memoryview((ctypes.c_char * CHUNK_SIZE_BYTES).from_address(buf_ptr.value))

    current_lba = 0
    total_bytes_read = 0

    t0 = time.perf_counter_ns()

    while total_bytes_read < test_limit_bytes and current_lba < total_lbas:
        byte_offset = current_lba * SECTOR_SIZE
        bytes_to_read = min(CHUNK_SIZE_BYTES, test_limit_bytes - total_bytes_read)
        
        bytes_read = os.preadv(fd, [memview[:bytes_to_read]], byte_offset)
        if bytes_read == 0:
            break

        current_lba += bytes_read // SECTOR_SIZE
        total_bytes_read += bytes_read

    tn = time.perf_counter_ns()
    libc.free(buf_ptr)

    elapsed_sec = (tn - t0) / 1e9
    throughput_mb = (total_bytes_read / (1024 * 1024)) / elapsed_sec if elapsed_sec > 0 else 0

    print(f"Finished reading {total_bytes_read / (1024**3):.2f} GB ({total_bytes_read:,} bytes).")
    print(f"Time Taken  : {elapsed_sec:.6f} seconds")
    print(f"Throughput  : {throughput_mb:.2f} MB/s\n")

finally:
    os.close(fd)


# =====================================================================
# 2. SEQUENTIAL READ - 5 GB (Parallel 4 KB Chunks by LBA Range)
# =====================================================================
print(f"--- Test 2: Sequential 5 GB Read in 4 KB Chunks ({NUM_WORKERS} Threads / QD={NUM_WORKERS}) ---")
CHUNK_SIZE_BYTES = 4096
LBAS_PER_CHUNK = CHUNK_SIZE_BYTES // SECTOR_SIZE

fd = os.open(BLOCK_DEVICE, flags)

try:
    total_bytes, total_lbas = get_device_capacity_lbas(fd, SECTOR_SIZE)
    test_limit_bytes = min(total_bytes, TARGET_BYTES_5GB)
    max_lbas = test_limit_bytes // SECTOR_SIZE
    
    sequential_lbas = list(range(0, max_lbas, LBAS_PER_CHUNK))

    def read_lba_4k(lba):
        byte_offset = lba * SECTOR_SIZE
        memview = get_thread_buffer(CHUNK_SIZE_BYTES)
        return os.preadv(fd, [memview], byte_offset)

    t0 = time.perf_counter_ns()

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(executor.map(read_lba_4k, sequential_lbas))

    tn = time.perf_counter_ns()

    total_bytes_read = sum(results)
    elapsed_sec = (tn - t0) / 1e9
    throughput_mb = (total_bytes_read / (1024 * 1024)) / elapsed_sec if elapsed_sec > 0 else 0
    iops = len(sequential_lbas) / elapsed_sec if elapsed_sec > 0 else 0

    print(f"Finished reading {total_bytes_read / (1024**3):.2f} GB in 4KB chunks across {len(sequential_lbas):,} LBAs.")
    print(f"Time Taken  : {elapsed_sec:.6f} seconds")
    print(f"Throughput  : {throughput_mb:.2f} MB/s")
    print(f"Read IOPS   : {iops:.2f} IOPS\n")

finally:
    os.close(fd)


# =====================================================================
# 3. SCATTERED RANDOM READ - 5 GB (Parallel Across Entire LBA Range)
# =====================================================================
print(f"--- Test 3: Scattered Random 5 GB Read in 4 KB Chunks ({NUM_WORKERS} Threads / QD={NUM_WORKERS}) ---")
CHUNK_SIZE_BYTES = 4096
LBAS_PER_CHUNK = CHUNK_SIZE_BYTES // SECTOR_SIZE
NUM_READS = TARGET_BYTES_5GB // CHUNK_SIZE_BYTES  # Exactly 1,310,720 reads to equal 5 GB

fd = os.open(BLOCK_DEVICE, flags)

try:
    total_bytes, total_lbas = get_device_capacity_lbas(fd, SECTOR_SIZE)
    max_chunk_index = (total_lbas - LBAS_PER_CHUNK) // LBAS_PER_CHUNK

    # Generate random LBA targets
    scattered_lbas = [random.randint(0, max_chunk_index) * LBAS_PER_CHUNK for _ in range(NUM_READS)]

    def read_random_lba_4k(lba):
        byte_offset = lba * SECTOR_SIZE
        memview = get_thread_buffer(CHUNK_SIZE_BYTES)
        return os.preadv(fd, [memview], byte_offset)

    t0 = time.perf_counter_ns()

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(executor.map(read_random_lba_4k, scattered_lbas))

    tn = time.perf_counter_ns()

    total_bytes_read = sum(results)
    elapsed_sec = (tn - t0) / 1e9
    throughput_mb = (total_bytes_read / (1024 * 1024)) / elapsed_sec if elapsed_sec > 0 else 0
    iops = len(scattered_lbas) / elapsed_sec if elapsed_sec > 0 else 0

    print(f"Completed   : {len(scattered_lbas):,} scattered reads ({total_bytes_read / (1024**3):.2f} GB total data)")
    print(f"Time Taken  : {elapsed_sec:.4f} seconds")
    print(f"Throughput  : {throughput_mb:.2f} MB/s")
    print(f"Random IOPS : {iops:.2f} IOPS\n")

finally:
    os.close(fd)


# =====================================================================
# 4. OVERHEAD MEASUREMENT EXPERIMENT - 5 GB Each (64 MB vs 4 KB)
# =====================================================================
print(f"--- Test 4: LBA Overhead Measurement (5 GB per test) ---")

def run_lba_overhead_test(device_path, chunk_size_bytes, num_workers, test_name):
    fd = os.open(device_path, flags)
    try:
        total_bytes, total_lbas = get_device_capacity_lbas(fd, SECTOR_SIZE)
        lbas_per_chunk = chunk_size_bytes // SECTOR_SIZE
        
        test_limit_bytes = min(total_bytes, TARGET_BYTES_5GB)
        max_lbas = test_limit_bytes // SECTOR_SIZE

        # 1. SETUP PHASE
        t_setup_start = time.perf_counter()
        lbas = list(range(0, max_lbas, lbas_per_chunk))
        num_requests = len(lbas)
        t_setup_end = time.perf_counter()
        setup_time = t_setup_end - t_setup_start

        individual_times = [0.0] * num_requests

        def read_lba_block(index_and_lba):
            idx, lba = index_and_lba
            byte_offset = lba * SECTOR_SIZE
            memview = get_thread_buffer(chunk_size_bytes)
            
            r_start = time.perf_counter_ns()
            bytes_read = os.preadv(fd, [memview], byte_offset)
            r_end = time.perf_counter_ns()
            
            individual_times[idx] = (r_end - r_start) / 1e6  # in ms
            return bytes_read

        # 2. EXECUTION PHASE
        cpu_start = os.times()
        t0 = time.perf_counter()

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(read_lba_block, enumerate(lbas)))

        tn = time.perf_counter()
        cpu_end = os.times()

        total_bytes_read = sum(results)
        wall_time = tn - t0
        
        user_cpu_time = cpu_end.user - cpu_start.user
        sys_cpu_time = cpu_end.system - cpu_start.system
        total_cpu_time = user_cpu_time + sys_cpu_time

        avg_request_latency_ms = sum(individual_times) / num_requests if num_requests > 0 else 0
        throughput_mb = (total_bytes_read / (1024 * 1024)) / wall_time if wall_time > 0 else 0

        print(f"\n[{test_name}]")
        print(f"  Total Requests Executed    : {num_requests:,}")
        print(f"  Total Data Read            : {total_bytes_read / (1024**3):.2f} GB")
        print(f"  Wall Clock Execution Time  : {wall_time:.4f} seconds")
        print(f"  Setup / Scheduling Time     : {setup_time * 1000:.2f} ms")
        print(f"  ------------------------------------------------")
        print(f"  Kernel Syscall CPU Time    : {sys_cpu_time:.4f} CPU seconds")
        print(f"  User Python CPU Time       : {user_cpu_time:.4f} CPU seconds")
        print(f"  Total CPU Time Consumed    : {total_cpu_time:.4f} CPU seconds")
        print(f"  CPU / Wall Efficiency Ratio: {(total_cpu_time / wall_time):.2f}x")
        print(f"  ------------------------------------------------")
        print(f"  Avg Latency Per Request    : {avg_request_latency_ms:.4f} ms")
        print(f"  Effective Throughput       : {throughput_mb:.2f} MB/s")

    finally:
        os.close(fd)


run_lba_overhead_test(BLOCK_DEVICE, 64 * 1024 * 1024, 1, "64 MB Chunks (Single Thread - 5 GB Total)")
run_lba_overhead_test(BLOCK_DEVICE, 4096, NUM_WORKERS, "4 KB Chunks (128 Threads - 5 GB Total)")