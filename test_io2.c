#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <liburing.h>

#define QUEUE_DEPTH 256
#define CHUNK_SIZE 4096
#define NUM_READS 100000

int main() {
    const char *filename = "/mnt/nvme/feat_l0_p9.pt";
    int fd = open(filename, O_RDONLY | O_DIRECT);
    if (fd < 0) { perror("open failed"); return 1; }

    off_t file_size = lseek(fd, 0, SEEK_END);
    size_t total_blocks = file_size / CHUNK_SIZE;

    struct io_uring ring;
    io_uring_queue_init(QUEUE_DEPTH, &ring, 0);

    // Allocate aligned memory buffers
    void *buffers[QUEUE_DEPTH];
    for (int i = 0; i < QUEUE_DEPTH; i++) {
        if (posix_memalign(&buffers[i], 4096, CHUNK_SIZE) != 0) {
            perror("Memory allocation failed");
            return 1;
        }
    }

    // Generate random offsets
    off_t *offsets = malloc(NUM_READS * sizeof(off_t));
    srand(time(NULL));
    for (int i = 0; i < NUM_READS; i++) {
        offsets[i] = (rand() % total_blocks) * CHUNK_SIZE;
    }

    printf("--- Running C Native io_uring Benchmark ---\n");
    printf("Requests: %d | Queue Depth: %d\n", NUM_READS, QUEUE_DEPTH);

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    int offset_idx = 0, completed = 0, in_flight = 0;

    while (completed < NUM_READS) {
        while (in_flight < QUEUE_DEPTH && offset_idx < NUM_READS) {
            struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
            if (!sqe) break;

            int buf_idx = offset_idx % QUEUE_DEPTH;
            io_uring_prep_read(sqe, fd, buffers[buf_idx], CHUNK_SIZE, offsets[offset_idx]);

            offset_idx++;
            in_flight++;
        }

        io_uring_submit(&ring);

        struct io_uring_cqe *cqe;
        int ret = io_uring_wait_cqe(&ring, &cqe);
        if (ret < 0) break;

        // Harvest all available completions
        unsigned head;
        unsigned count = 0;
        io_uring_for_each_cqe(&ring, head, cqe) {
            count++;
        }
        io_uring_cq_advance(&ring, count);

        completed += count;
        in_flight -= count;
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double elapsed = (t_end.tv_sec - t_start.tv_sec) + (t_end.tv_nsec - t_start.tv_nsec) / 1e9;

    double mb = ((double)NUM_READS * CHUNK_SIZE) / (1024.0 * 1024.0);
    printf("\nDone in %.4f seconds\n", elapsed);
    printf("Throughput  : %.2f MB/s\n", mb / elapsed);
    printf("Random IOPS : %.2f IOPS\n", NUM_READS / elapsed);

    io_uring_queue_exit(&ring);
    close(fd);
    return 0;
}