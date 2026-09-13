import os
import time
import ctypes
import random
import pyuring

ALIGNMENT = 4096
ENTRIES = 256         # Queue Depth (Ring Size)
BATCH_SIZE = 128      # Batch submission size
CHUNK_SIZE = 4096     # 4 KB
FILENAME = "/mnt/nvme/feat_l0_p9.pt"

# Allocated C-aligned buffer pool for io_uring operations
def allocate_aligned_buffer(size):
    buf_ptr = ctypes.c_void_p()
    libc = ctypes.CDLL(None)
    if libc.posix_memalign(ctypes.byref(buf_ptr), ALIGNMENT, size) != 0:
        raise OSError("posix_memalign failed")
    return buf_ptr, memoryview((ctypes.c_char * size).from_address(buf_ptr.value))


# =====================================================================
# IO_URING HIGH-THROUGHPUT RANDOM 4 KB READS
# =====================================================================
def run_io_uring_benchmark():
    flags = os.O_RDONLY | os.O_DIRECT
    fd = os.open(FILENAME, flags)

    try:
        file_size = os.lseek(fd, 0, os.SEEK_END)
        total_blocks = file_size // CHUNK_SIZE
        
        NUM_READS = min(100000, total_blocks)  # Issue up to 100k random reads
        scattered_offsets = [random.randrange(0, total_blocks) * CHUNK_SIZE for _ in range(NUM_READS)]

        print(f"--- Running io_uring Async Benchmark ---")
        print(f"File Size        : {file_size / (1024**3):.2f} GB")
        print(f"Total Requests   : {NUM_READS:,} 4KB reads")
        print(f"Target Queue Depth: {ENTRIES}")

        # Initialize single buffer pool for in-flight requests
        # We need ENTRIES worth of buffers for the queue
        buffers = []
        iov_ptrs = []
        for _ in range(ENTRIES):
            ptr, mv = allocate_aligned_buffer(CHUNK_SIZE)
            buffers.append((ptr, mv))

        ring = pyuring.ring()
        pyuring.io_uring_queue_init(ENTRIES, ring, 0)

        offset_idx = 0
        completed = 0
        in_flight = 0

        t0 = time.perf_counter()

        while completed < NUM_READS:
            # 1. FILL SUBMISSION QUEUE (SQ) UP TO BATCH SIZE
            while in_flight < ENTRIES and offset_idx < NUM_READS:
                sqe = pyuring.io_uring_get_sqe(ring)
                if not sqe:
                    break

                buf_idx = in_flight % ENTRIES
                ptr, mv = buffers[buf_idx]
                offset = scattered_offsets[offset_idx]

                # Prepare preadv / read operation in the submission queue
                pyuring.io_uring_prep_read(sqe, fd, ptr, CHUNK_SIZE, offset)
                
                offset_idx += 1
                in_flight += 1

            # 2. SUBMIT BATCH TO NVME HARDWARE VIA SINGLE SYSCALL
            pyuring.io_uring_submit(ring)

            # 3. REAP COMPLETIONS (CQ)
            cqe = pyuring.cqe()
            # Wait for at least 1 completion before loop continues
            pyuring.io_uring_wait_cqe(ring, cqe)

            while cqe:
                completed += 1
                in_flight -= 1
                pyuring.io_uring_cqe_seen(ring, cqe)
                
                # Check for additional completions ready in ring buffer without blocking
                try:
                    pyuring.io_uring_peek_cqe(ring, cqe)
                except Exception:
                    break

        t1 = time.perf_counter()

        elapsed_sec = t1 - t0
        total_bytes = completed * CHUNK_SIZE
        throughput_mb = (total_bytes / (1024 * 1024)) / elapsed_sec
        iops = completed / elapsed_sec

        print(f"\n[Results]")
        print(f"  Time Taken  : {elapsed_sec:.4f} seconds")
        print(f"  Throughput  : {throughput_mb:.2f} MB/s")
        print(f"  Random IOPS : {iops:,.2f} IOPS")

        pyuring.io_uring_queue_exit(ring)

    finally:
        os.close(fd)


if __name__ == "__main__":
    run_io_uring_benchmark()