import os
import time
import ctypes
import random
import libaio

ALIGNMENT = 4096
ENTRIES = 256
CHUNK_SIZE = 4096
FILENAME = "/mnt/nvme/feat_l0_p9.pt"

def allocate_aligned_buffer(size):
    buf_ptr = ctypes.c_void_p()
    libc = ctypes.CDLL(None)
    if libc.posix_memalign(ctypes.byref(buf_ptr), ALIGNMENT, size) != 0:
        raise OSError("posix_memalign failed")
    return buf_ptr

def run_libaio_benchmark():
    flags = os.O_RDONLY | os.O_DIRECT
    fd = os.open(FILENAME, flags)

    try:
        file_size = os.lseek(fd, 0, os.SEEK_END)
        total_blocks = file_size // CHUNK_SIZE
        
        NUM_READS = min(100000, total_blocks)
        scattered_offsets = [random.randrange(0, total_blocks) * CHUNK_SIZE for _ in range(NUM_READS)]

        print(f"--- Running Async Kernel AIO Benchmark ---")
        print(f"File Size         : {file_size / (1024**3):.2f} GB")
        print(f"Total Requests    : {NUM_READS:,} 4KB reads")
        print(f"Target Queue Depth: {ENTRIES}")

        # Allocate memory buffers for in-flight requests
        buffers = [allocate_aligned_buffer(CHUNK_SIZE) for _ in range(ENTRIES)]
        
        ctx = libaio.AIOContext(ENTRIES)

        offset_idx = 0
        completed = 0
        in_flight = 0

        t0 = time.perf_counter()

        while completed < NUM_READS:
            # 1. Fill queue with requests
            iocbs = []
            while in_flight < ENTRIES and offset_idx < NUM_READS:
                buf_idx = offset_idx % ENTRIES
                buf_ptr = buffers[buf_idx]
                offset = scattered_offsets[offset_idx]

                # Create asynchronous read control block
                block = ctypes.create_string_buffer(CHUNK_SIZE)
                iocb = libaio.AIOBlock(
                    mode=libaio.AIOCMD_PREAD,
                    fd=fd,
                    buf=buf_ptr.value,
                    count=CHUNK_SIZE,
                    offset=offset
                )
                iocbs.append(iocb)
                offset_idx += 1
                in_flight += 1

            # 2. Submit batch directly to kernel driver
            if iocbs:
                ctx.submit(iocbs)

            # 3. Harvest completions
            events = ctx.getevents(min_nr=1, nr=ENTRIES)
            num_events = len(events)
            completed += num_events
            in_flight -= num_events

        t1 = time.perf_counter()

        elapsed_sec = t1 - t0
        total_bytes = completed * CHUNK_SIZE
        throughput_mb = (total_bytes / (1024 * 1024)) / elapsed_sec
        iops = completed / elapsed_sec

        print(f"\n[Results]")
        print(f"  Time Taken  : {elapsed_sec:.4f} seconds")
        print(f"  Throughput  : {throughput_mb:.2f} MB/s")
        print(f"  Random IOPS : {iops:,.2f} IOPS")

    finally:
        os.close(fd)

if __name__ == "__main__":
    run_libaio_benchmark()