import os
import ctypes
import time
import random

FILENAME = "/mnt/nvme/feat_l0_p0.pt"
ALIGNMENT = 4096


# 1. FULL FILE
flags = os.O_RDONLY | os.O_DIRECT
fd = os.open(FILENAME, flags)
FILE_SIZE = os.lseek(fd, 0, os.SEEK_END)

try:
    t0 = time.perf_counter_ns()
    libc = ctypes.CDLL(None)
    buf_ptr = ctypes.c_void_p()

    result = libc.posix_memalign(ctypes.byref(buf_ptr), ALIGNMENT, ALIGNMENT)
    if result != 0:
        raise OSError(result, "posix_memalign failed")
    
    aligned_buffer = (ctypes.c_char * ALIGNMENT).from_address(buf_ptr.value)

    bytes_read = os.readinto(fd, aligned_buffer)
    print(f"Successfully read {bytes_read} bytes using O_DIRECT.")

    data = bytes(aligned_buffer.raw[:bytes_read])
    print(f"Read sample data (first 10 bytes): {data[:10]}")

    tn = time.perf_counter_ns()

    print(f"Read full file in = {(tn-t0)/1024**3}")

    libc.free(buf_ptr)

finally:
    os.close(fd)



# 2. Partial reads for full file in same order (sequential)
flags = os.O_RDONLY | os.O_DIRECT
fd = os.open(FILENAME, flags)

try:
    # Determine total file size
    file_size = os.lseek(fd, 0, os.SEEK_END)
    os.lseek(fd, 0, os.SEEK_SET)

    # Allocate a single 4096-byte aligned memory buffer
    libc = ctypes.CDLL(None)
    buf_ptr = ctypes.c_void_p()

    result = libc.posix_memalign(ctypes.byref(buf_ptr), ALIGNMENT, 4096)
    if result != 0:
        raise OSError(result, "posix_memalign failed")

    aligned_buffer = (ctypes.c_char * 4096).from_address(buf_ptr.value)

    offset = 0
    total_bytes_read = 0

    # Start timer right before the actual reading loop
    t0 = time.perf_counter_ns()

    while offset < file_size:
        # Explicit offset read via os.preadinto
        bytes_read = os.preadinto(fd, aligned_buffer, offset)

        if bytes_read == 0:
            break  # EOF reached

        # Extract/process chunk data if needed
        chunk_data = bytes(aligned_buffer.raw[:bytes_read])

        offset += bytes_read
        total_bytes_read += bytes_read

    tn = time.perf_counter_ns()

    # Free memory buffer
    libc.free(buf_ptr)

    # Performance output
    elapsed_sec = (tn - t0) / 1e9  # Convert nanoseconds to seconds
    throughput_mb = (total_bytes_read / (1024 * 1024)) / elapsed_sec if elapsed_sec > 0 else 0

    print(f"Finished reading. Total bytes read: {total_bytes_read} / {file_size}")
    print(f"Read full file in 4KB chunks: {elapsed_sec:.6f} seconds ({throughput_mb:.2f} MB/s)")

finally:
    os.close(fd)



# 3. Partial reads with only taget features --> set to 5% scattered in file
ALIGNMENT = 4096
CHUNK_SIZE = 4096
NUM_READS = 13700

flags = os.O_RDONLY | os.O_DIRECT
fd = os.open(FILENAME, flags)

try:
    # 1. Get total file size dynamically
    file_size = os.lseek(fd, 0, os.SEEK_END)
    os.lseek(fd, 0, os.SEEK_SET)

    # 2. Compute total available 4 KB block indices
    total_blocks = file_size // CHUNK_SIZE
    if total_blocks == 0:
        raise ValueError("File is too small for 4 KB block aligned reads.")

    # 3. Generate exactly 13,700 block-aligned offsets across the entire file
    # random.randrange(0, total_blocks) ensures no out-of-bounds offsets
    mapping = [random.randrange(0, total_blocks) * CHUNK_SIZE for _ in range(NUM_READS)]

    # 4. Allocate a single 4 KB aligned memory buffer for reuse
    libc = ctypes.CDLL(None)
    buf_ptr = ctypes.c_void_p()

    result = libc.posix_memalign(ctypes.byref(buf_ptr), ALIGNMENT, CHUNK_SIZE)
    if result != 0:
        raise OSError(result, "posix_memalign failed")

    aligned_buffer = (ctypes.c_char * CHUNK_SIZE).from_address(buf_ptr.value)

    total_bytes_read = 0

    # 5. Measure random direct I/O read latency/throughput
    t0 = time.perf_counter_ns()

    for offset in mapping:
        bytes_read = os.preadinto(fd, aligned_buffer, offset)

        if bytes_read == 0:
            continue  # Safety check in case of unexpected EOF

        total_bytes_read += bytes_read

    tn = time.perf_counter_ns()

    # Free allocated buffer
    libc.free(buf_ptr)

    # 6. Performance calculation
    elapsed_sec = (tn - t0) / 1e9
    throughput_mb = (total_bytes_read / (1024 * 1024)) / elapsed_sec if elapsed_sec > 0 else 0
    iops = len(mapping) / elapsed_sec if elapsed_sec > 0 else 0

    print(f"File Size: {file_size / (1024**3):.2f} GB ({total_blocks:,} total blocks)")
    print(f"Completed {len(mapping):,} scattered reads ({total_bytes_read / (1024**2):.2f} MB total data).")
    print(f"Time Taken  : {elapsed_sec:.4f} seconds")
    print(f"Throughput  : {throughput_mb:.2f} MB/s")
    print(f"Random IOPS : {iops:.2f} IOPS")

finally:
    os.close(fd)