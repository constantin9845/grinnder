import os
import time
import ctypes
import random
import pyuring

ALIGNMENT = 4096
ENTRIES = 256         # Queue Depth
CHUNK_SIZE = 4096     # 4 KB
FILENAME = "/mnt/nvme/feat_l0_p9.pt"

def allocate_aligned_buffer(size):
    buf_ptr = ctypes.c_void_p()
    libc = ctypes.CDLL(None)
    if libc.posix_memalign(ctypes.byref(buf_ptr), ALIGNMENT, size) != 0:
        raise OSError("posix_memalign failed")
    return buf_ptr


def run_io_uring_benchmark():
    flags = os.O_RDONLY | os.O_DIRECT
    fd = os.open(FILENAME, flags)

    try:
        file_size = os.lseek(fd, 0, os.SEEK_END)
        total_blocks = file_size // CHUNK_SIZE
        
        NUM_READS = min(100000, total_blocks)
        scattered_offsets = [random.randrange(0, total_blocks) * CHUNK_SIZE for _ in range(NUM_READS)]

        print(f"--- Running io_uring Async Benchmark ---")
        print(f"File Size         : {file_size / (1024**3):.2f} GB")
        print(f"Total Requests    : {NUM_READS:,} 4KB reads")
        print(f"Target Queue Depth: {ENTRIES}")

        # Allocate buffer pool for in-flight requests
        buffer_ptrs = [allocate_aligned_buffer(CHUNK_SIZE) for _ in range(ENTRIES)]

        # 1. Correct class initialization in pyuring
        ring = pyuring.queue()
        ring.ring_init(ENTRIES, 0)

        cqes = pyuring.cqes()

        offset_idx = 0
        completed = 0
        in_flight = 0

        t0 = time.perf_counter()

        while completed < NUM_READS:
            # Fill Submission Queue (SQ)
            while in_flight < ENTRIES and offset_idx < NUM_READS:
                sqe = ring.get_sqe()
                if not sqe:
                    break

                buf_idx = offset_idx % ENTRIES
                buf_ptr = buffer_ptrs[buf_idx]
                offset = scattered_offsets[offset_idx]

                # Prepare read operation
                pyuring.io_uring_prep_read(sqe, fd, buf_ptr, CHUNK_SIZE, offset)
                
                offset_idx += 1
                in_flight += 1

            # Submit batch to NVMe controller
            ring.submit()

            # Wait for at least 1 completion
            ring.wait_cqe(cqes)

            # Process completed requests
            for cqe in cqes:
                completed += 1
                in_flight -= 1
                ring.cqe_seen(cqe)

        t1 = time.perf_counter()

        elapsed_sec = t1 - t0
        total_bytes = completed * CHUNK_SIZE
        throughput_mb = (total_bytes / (1024 * 1024)) / elapsed_sec
        iops = completed / elapsed_sec

        print(f"\n[Results]")
        print(f"  Time Taken  : {elapsed_sec:.4f} seconds")
        print(f"  Throughput  : {throughput_mb:.2f} MB/s")
        print(f"  Random IOPS : {iops:,.2f} IOPS")

        ring.queue_exit()

    finally:
        os.close(fd)


if __name__ == "__main__":
    run_io_uring_benchmark()