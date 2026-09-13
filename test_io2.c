#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
#include <sys/resource.h>
#include <liburing.h>

#define QD_SMALL 256          // Queue Depth for 4KB Reads
#define QD_LARGE 16           // Queue Depth for 2MB Large Chunk Reads
#define SECTOR_4K 4096        // 4 KB Sector
#define CHUNK_2M (2 * 1024 * 1024) // 2 MB Chunk

// High-precision time helper (seconds)
static double get_time_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return ts.tv_sec + (ts.tv_nsec / 1e9);
}

// CPU usage helper (Kernel + User CPU seconds)
static double get_cpu_time_sec() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    double user = ru.ru_utime.tv_sec + (ru.ru_utime.tv_usec / 1e6);
    double sys  = ru.ru_stime.tv_sec + (ru.ru_stime.tv_usec / 1e6);
    return user + sys;
}

void print_stats(const char *title, size_t total_bytes, size_t num_reads, 
                 double wall_time, double prep_time, double wait_time, 
                 double cpu_time) {
    double mb = (double)total_bytes / (1024.0 * 1024.0);
    double throughput = (wall_time > 0) ? (mb / wall_time) : 0;
    double iops = (wall_time > 0) ? ((double)num_reads / wall_time) : 0;
    
    double host_pct = (wall_time > 0) ? ((prep_time / wall_time) * 100.0) : 0;
    double ssd_pct = (wall_time > 0) ? ((wait_time / wall_time) * 100.0) : 0;

    printf("=====================================================\n");
    printf(" %s\n", title);
    printf("=====================================================\n");
    printf("  Total Data Processed : %.2f MB (%zu requests)\n", mb, num_reads);
    printf("  Wall Clock Run Time  : %.4f seconds\n", wall_time);
    printf("  Throughput           : %.2f MB/s\n", throughput);
    printf("  IOPS                 : %.2f IOPS\n", iops);
    printf("  ---------------------------------------------------\n");
    printf("  Total CPU Time Used  : %.4f CPU sec (CPU Load: %.1f%%)\n", 
           cpu_time, (wall_time > 0) ? ((cpu_time / wall_time) * 100.0) : 0);
    printf("  Host Submission Time : %.4f sec (%.1f%% of wall time)\n", prep_time, host_pct);
    printf("  SSD Hardware Wait    : %.4f sec (%.1f%% of wall time)\n", wait_time, ssd_pct);
    printf("\n");
}


// =====================================================================
// TEST 1: FIXED FULL FILE READ (2 MB Chunks via io_uring, QD=16)
// =====================================================================
void run_test1_full_2m_uring(const char *filename) {
    int fd = open(filename, O_RDONLY | O_DIRECT);
    if (fd < 0) { perror("Test 1 Open Failed"); return; }

    off_t file_size = lseek(fd, 0, SEEK_END);
    size_t num_reads = (file_size + CHUNK_2M - 1) / CHUNK_2M;

    struct io_uring ring;
    if (io_uring_queue_init(QD_LARGE, &ring, 0) < 0) {
        perror("Test 1 io_uring_queue_init failed");
        close(fd);
        return;
    }

    void *buffers[QD_LARGE];
    for (int i = 0; i < QD_LARGE; i++) {
        if (posix_memalign(&buffers[i], 4096, CHUNK_2M) != 0) {
            perror("Test 1 posix_memalign Failed");
            close(fd);
            return;
        }
    }

    double cpu_start = get_cpu_time_sec();
    double t_start = get_time_sec();

    size_t completed = 0, in_flight = 0, current_idx = 0;
    off_t current_offset = 0;

    double total_prep_time = 0.0;
    double total_wait_time = 0.0;

    while (completed < num_reads) {
        double prep_t0 = get_time_sec();

        while (in_flight < QD_LARGE && current_idx < num_reads) {
            struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
            if (!sqe) break;

            int buf_idx = current_idx % QD_LARGE;
            size_t bytes_to_read = (file_size - current_offset > CHUNK_2M) 
                                   ? CHUNK_2M 
                                   : (file_size - current_offset);

            io_uring_prep_read(sqe, fd, buffers[buf_idx], bytes_to_read, current_offset);
            
            current_offset += bytes_to_read;
            current_idx++;
            in_flight++;
        }

        double prep_t1 = get_time_sec();
        total_prep_time += (prep_t1 - prep_t0);

        double wait_t0 = get_time_sec();
        io_uring_submit(&ring);

        struct io_uring_cqe *cqe;
        int ret = io_uring_wait_cqe(&ring, &cqe);
        if (ret < 0) break;

        unsigned head;
        unsigned count = 0;
        io_uring_for_each_cqe(&ring, head, cqe) {
            count++;
        }
        io_uring_cq_advance(&ring, count);

        completed += count;
        in_flight -= count;

        double wait_t1 = get_time_sec();
        total_wait_time += (wait_t1 - wait_t0);
    }

    double t_end = get_time_sec();
    double cpu_end = get_cpu_time_sec();

    print_stats("TEST 1: Full File Read (2MB Chunks via io_uring QD=16)", 
                file_size, num_reads, t_end - t_start, 
                total_prep_time, total_wait_time, cpu_end - cpu_start);

    for (int i = 0; i < QD_LARGE; i++) free(buffers[i]);
    io_uring_queue_exit(&ring);
    close(fd);
}


// =====================================================================
// TEST 2: RANDOM 4 KB READS (10% of File Size via io_uring QD=256)
// =====================================================================
void run_test2_random_4k(const char *filename) {
    int fd = open(filename, O_RDONLY | O_DIRECT);
    if (fd < 0) { perror("Test 2 Open Failed"); return; }

    off_t file_size = lseek(fd, 0, SEEK_END);
    size_t total_4k_blocks = file_size / SECTOR_4K;
    size_t num_reads = total_4k_blocks / 10; // 10% of file size

    if (num_reads == 0) num_reads = 1;

    off_t *offsets = malloc(num_reads * sizeof(off_t));
    if (!offsets) {
        perror("Test 2 malloc failed");
        close(fd);
        return;
    }

    srand(42); // Fixed seed for reproducibility
    for (size_t i = 0; i < num_reads; i++) {
        offsets[i] = (rand() % total_4k_blocks) * SECTOR_4K;
    }

    struct io_uring ring;
    if (io_uring_queue_init(QD_SMALL, &ring, 0) < 0) {
        perror("Test 2 io_uring_queue_init failed");
        free(offsets);
        close(fd);
        return;
    }

    void *buffers[QD_SMALL];
    for (int i = 0; i < QD_SMALL; i++) {
        if (posix_memalign(&buffers[i], 4096, SECTOR_4K) != 0) {
            perror("Test 2 posix_memalign Failed");
            free(offsets);
            close(fd);
            return;
        }
    }

    double cpu_start = get_cpu_time_sec();
    double t_start = get_time_sec();

    size_t offset_idx = 0, completed = 0, in_flight = 0;
    double total_prep_time = 0.0;
    double total_wait_time = 0.0;

    while (completed < num_reads) {
        double prep_t0 = get_time_sec();

        while (in_flight < QD_SMALL && offset_idx < num_reads) {
            struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
            if (!sqe) break;

            int buf_idx = offset_idx % QD_SMALL;
            io_uring_prep_read(sqe, fd, buffers[buf_idx], SECTOR_4K, offsets[offset_idx]);
            
            offset_idx++;
            in_flight++;
        }

        double prep_t1 = get_time_sec();
        total_prep_time += (prep_t1 - prep_t0);

        double wait_t0 = get_time_sec();
        io_uring_submit(&ring);

        struct io_uring_cqe *cqe;
        int ret = io_uring_wait_cqe(&ring, &cqe);
        if (ret < 0) break;

        unsigned head;
        unsigned count = 0;
        io_uring_for_each_cqe(&ring, head, cqe) {
            count++;
        }
        io_uring_cq_advance(&ring, count);

        completed += count;
        in_flight -= count;

        double wait_t1 = get_time_sec();
        total_wait_time += (wait_t1 - wait_t0);
    }

    double t_end = get_time_sec();
    double cpu_end = get_cpu_time_sec();

    print_stats("TEST 2: Random 4KB Reads (10% of File Size via io_uring QD=256)", 
                completed * SECTOR_4K, num_reads, t_end - t_start, 
                total_prep_time, total_wait_time, cpu_end - cpu_start);

    for (int i = 0; i < QD_SMALL; i++) free(buffers[i]);
    free(offsets);
    io_uring_queue_exit(&ring);
    close(fd);
}


// =====================================================================
// TEST 3: FULL FILE SEQUENTIAL 4 KB READS (via io_uring QD=256)
// =====================================================================
void run_test3_seq_4k(const char *filename) {
    int fd = open(filename, O_RDONLY | O_DIRECT);
    if (fd < 0) { perror("Test 3 Open Failed"); return; }

    off_t file_size = lseek(fd, 0, SEEK_END);
    size_t num_reads = file_size / SECTOR_4K;

    struct io_uring ring;
    if (io_uring_queue_init(QD_SMALL, &ring, 0) < 0) {
        perror("Test 3 io_uring_queue_init failed");
        close(fd);
        return;
    }

    void *buffers[QD_SMALL];
    for (int i = 0; i < QD_SMALL; i++) {
        if (posix_memalign(&buffers[i], 4096, SECTOR_4K) != 0) {
            perror("Test 3 posix_memalign Failed");
            close(fd);
            return;
        }
    }

    double cpu_start = get_cpu_time_sec();
    double t_start = get_time_sec();

    size_t completed = 0, in_flight = 0, current_idx = 0;
    off_t current_offset = 0;

    double total_prep_time = 0.0;
    double total_wait_time = 0.0;

    while (completed < num_reads) {
        double prep_t0 = get_time_sec();

        while (in_flight < QD_SMALL && current_idx < num_reads) {
            struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
            if (!sqe) break;

            int buf_idx = current_idx % QD_SMALL;
            io_uring_prep_read(sqe, fd, buffers[buf_idx], SECTOR_4K, current_offset);
            
            current_offset += SECTOR_4K;
            current_idx++;
            in_flight++;
        }

        double prep_t1 = get_time_sec();
        total_prep_time += (prep_t1 - prep_t0);

        double wait_t0 = get_time_sec();
        io_uring_submit(&ring);

        struct io_uring_cqe *cqe;
        int ret = io_uring_wait_cqe(&ring, &cqe);
        if (ret < 0) break;

        unsigned head;
        unsigned count = 0;
        io_uring_for_each_cqe(&ring, head, cqe) {
            count++;
        }
        io_uring_cq_advance(&ring, count);

        completed += count;
        in_flight -= count;

        double wait_t1 = get_time_sec();
        total_wait_time += (wait_t1 - wait_t0);
    }

    double t_end = get_time_sec();
    double cpu_end = get_cpu_time_sec();

    print_stats("TEST 3: Full File Sequential 4KB Reads (via io_uring QD=256)", 
                completed * SECTOR_4K, num_reads, t_end - t_start, 
                total_prep_time, total_wait_time, cpu_end - cpu_start);

    for (int i = 0; i < QD_SMALL; i++) free(buffers[i]);
    io_uring_queue_exit(&ring);
    close(fd);
}


// =====================================================================
// MAIN ENTRY
// =====================================================================
int main(int argc, char *argv[]) {
    const char *filename1 = "/mnt/nvme/feat_l0_p6.pt";
    const char *filename2 = "/mnt/nvme/feat_l0_p7.pt";
    const char *filename3 = "/mnt/nvme/feat_l0_p8.pt";
    if (argc > 1) {
        filename1 = argv[1];
    }

    printf("\nStarting Fixed Benchmark Suite: %s\n\n");

    run_test1_full_2m_uring(filename1);
    run_test2_random_4k(filename2);
    run_test3_seq_4k(filename3);

    return 0;
}