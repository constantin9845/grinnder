#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
#include <sys/resource.h>
#include <cuda_runtime.h>
#include <cufile.h>

#define SECTOR_4K      (4 * 1024)         // 4 KB Sector Alignment
#define CHUNK_2M       (2 * 1024 * 1024)  // 2 MB Read Chunk
#define MAX_BATCH_SIZE 4096               // GDS API Limit for cuFileBatchIOSetUp
#define GPU_ID         0                  // Target GPU index

static double get_time_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return ts.tv_sec + (ts.tv_nsec / 1e9);
}

static double get_cpu_time_sec() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    double user = ru.ru_utime.tv_sec + (ru.ru_utime.tv_usec / 1e6);
    double sys  = ru.ru_stime.tv_sec + (ru.ru_stime.tv_usec / 1e6);
    return user + sys;
}

#define CUDA_CHECK(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
                (int)result, cudaGetErrorString(result), func);
        exit(EXIT_FAILURE);
    }
}

#define CUFILE_CHECK(status) check_cufile((status), #status, __FILE__, __LINE__)
void check_cufile(CUfileError_t status, char const *const func, const char *const file, int const line) {
    if (status.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "cuFile error at %s:%d code=%d \"%s\" \n", file, line,
                status.err, func);
        exit(EXIT_FAILURE);
    }
}

void print_stats(const char *title, size_t total_bytes, size_t num_reads, 
                 double wall_time, double cpu_time) {
    double mb = (double)total_bytes / (1024.0 * 1024.0);
    double throughput = (wall_time > 0) ? (mb / wall_time) : 0;
    double iops = (wall_time > 0) ? ((double)num_reads / wall_time) : 0;

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
    printf("\n");
}


// =====================================================================
// TEST 1: FULL FILE SEQUENTIAL READ (cuFile 2MB Chunks directly to GPU)
// =====================================================================
void run_test1_gds_2m_chunks(const char *filename) {
    int fd = open(filename, O_RDONLY | O_DIRECT);
    if (fd < 0) { perror("Test 1 Open Failed"); return; }

    off_t file_size = lseek(fd, 0, SEEK_END);
    if (file_size <= 0) { close(fd); return; }

    CUfileHandle_t cf_handle;
    CUfileDescr_t cf_descr;
    memset(&cf_descr, 0, sizeof(CUfileDescr_t));
    cf_descr.handle.fd = fd;
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    CUFILE_CHECK(cuFileHandleRegister(&cf_handle, &cf_descr));

    void *d_buffer = NULL;
    CUDA_CHECK(cudaMalloc(&d_buffer, CHUNK_2M));
    CUFILE_CHECK(cuFileBufRegister(d_buffer, CHUNK_2M, 0));

    double cpu_start = get_cpu_time_sec();
    double t_start = get_time_sec();

    off_t offset = 0;
    size_t num_reads = 0;

    while (offset < file_size) {
        size_t bytes_to_read = (file_size - offset > CHUNK_2M) 
                               ? CHUNK_2M 
                               : (file_size - offset);

        ssize_t ret = cuFileRead(cf_handle, d_buffer, bytes_to_read, offset, 0);
        if (ret < 0) {
            fprintf(stderr, "Test 1 GDS Read Error at offset %ld\n", offset);
            break;
        }

        offset += ret;
        num_reads++;
    }

    double t_end = get_time_sec();
    double cpu_end = get_cpu_time_sec();

    print_stats("GDS TEST 1: Full File Read (cuFile Direct NVMe->GPU, 2MB Chunks)", 
                file_size, num_reads, t_end - t_start, cpu_end - cpu_start);

    CUFILE_CHECK(cuFileBufDeregister(d_buffer));
    CUDA_CHECK(cudaFree(d_buffer));
    cuFileHandleDeregister(cf_handle);
    close(fd);
}


// =====================================================================
// TEST 2: SCATTERED 4 KB READS (cuFile Batch Async NVMe->GPU)
// =====================================================================
void run_test2_gds_random_4k(const char *filename) {
    int fd = open(filename, O_RDONLY | O_DIRECT);
    if (fd < 0) { perror("Test 2 Open Failed"); return; }

    off_t file_size = lseek(fd, 0, SEEK_END);
    size_t total_4k_blocks = file_size / SECTOR_4K;
    size_t num_reads = total_4k_blocks / 10;
    if (num_reads == 0) num_reads = 1;

    CUfileHandle_t cf_handle;
    CUfileDescr_t cf_descr;
    memset(&cf_descr, 0, sizeof(CUfileDescr_t));
    cf_descr.handle.fd = fd;
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    CUFILE_CHECK(cuFileHandleRegister(&cf_handle, &cf_descr));

    size_t gpu_buf_size = num_reads * SECTOR_4K;
    void *d_buffer = NULL;
    CUDA_CHECK(cudaMalloc(&d_buffer, gpu_buf_size));
    CUFILE_CHECK(cuFileBufRegister(d_buffer, gpu_buf_size, 0));

    // Prepare full params list
    CUfileIOParams_t *io_params = calloc(num_reads, sizeof(CUfileIOParams_t));
    srand(42);
    for (size_t i = 0; i < num_reads; i++) {
        io_params[i].mode = CUFILE_BATCH;
        io_params[i].opcode = CUFILE_READ;
        io_params[i].fh = cf_handle;
        io_params[i].u.batch.devPtr_base = d_buffer;
        io_params[i].u.batch.devPtr_offset = i * SECTOR_4K;
        io_params[i].u.batch.file_offset = (rand() % total_4k_blocks) * SECTOR_4K;
        io_params[i].u.batch.size = SECTOR_4K;
    }

    // Allocate temporary workspace for max batch size (4096)
    CUfileBatchHandle_t batch_handle;
    CUFILE_CHECK(cuFileBatchIOSetUp(&batch_handle, MAX_BATCH_SIZE));
    CUfileIOEvents_t events[MAX_BATCH_SIZE];

    double cpu_start = get_cpu_time_sec();
    double t_start = get_time_sec();

    size_t total_completed = 0;

    // Process in chunks of MAX_BATCH_SIZE (4096)
    for (size_t offset_idx = 0; offset_idx < num_reads; offset_idx += MAX_BATCH_SIZE) {
        unsigned int current_batch_size = (num_reads - offset_idx > MAX_BATCH_SIZE) 
                                          ? MAX_BATCH_SIZE 
                                          : (unsigned int)(num_reads - offset_idx);

        CUFILE_CHECK(cuFileBatchIOSubmit(batch_handle, current_batch_size, &io_params[offset_idx], 0));

        unsigned int completed = current_batch_size;
        CUFILE_CHECK(cuFileBatchIOGetStatus(batch_handle, current_batch_size, &completed, events, NULL));
        total_completed += completed;
    }

    double t_end = get_time_sec();
    double cpu_end = get_cpu_time_sec();

    print_stats("GDS TEST 2: Random 4KB Reads (10% File Size via cuFile Batch Async)", 
                total_completed * SECTOR_4K, num_reads, t_end - t_start, cpu_end - cpu_start);

    cuFileBatchIODestroy(batch_handle);
    free(io_params);
    CUFILE_CHECK(cuFileBufDeregister(d_buffer));
    CUDA_CHECK(cudaFree(d_buffer));
    cuFileHandleDeregister(cf_handle);
    close(fd);
}


// =====================================================================
// TEST 3: FULL FILE SEQUENTIAL 4 KB READS (cuFile Batch Async NVMe->GPU)
// =====================================================================
void run_test3_gds_seq_4k(const char *filename) {
    int fd = open(filename, O_RDONLY | O_DIRECT);
    if (fd < 0) { perror("Test 3 Open Failed"); return; }

    off_t file_size = lseek(fd, 0, SEEK_END);
    size_t num_reads = file_size / SECTOR_4K;

    CUfileHandle_t cf_handle;
    CUfileDescr_t cf_descr;
    memset(&cf_descr, 0, sizeof(CUfileDescr_t));
    cf_descr.handle.fd = fd;
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    CUFILE_CHECK(cuFileHandleRegister(&cf_handle, &cf_descr));

    size_t gpu_buf_size = num_reads * SECTOR_4K;
    void *d_buffer = NULL;
    CUDA_CHECK(cudaMalloc(&d_buffer, gpu_buf_size));
    CUFILE_CHECK(cuFileBufRegister(d_buffer, gpu_buf_size, 0));

    CUfileIOParams_t *io_params = calloc(num_reads, sizeof(CUfileIOParams_t));
    for (size_t i = 0; i < num_reads; i++) {
        io_params[i].mode = CUFILE_BATCH;
        io_params[i].opcode = CUFILE_READ;
        io_params[i].fh = cf_handle;
        io_params[i].u.batch.devPtr_base = d_buffer;
        io_params[i].u.batch.devPtr_offset = i * SECTOR_4K;
        io_params[i].u.batch.file_offset = i * SECTOR_4K;
        io_params[i].u.batch.size = SECTOR_4K;
    }

    // Allocate temporary workspace for max batch size (4096)
    CUfileBatchHandle_t batch_handle;
    CUFILE_CHECK(cuFileBatchIOSetUp(&batch_handle, MAX_BATCH_SIZE));
    CUfileIOEvents_t events[MAX_BATCH_SIZE];

    double cpu_start = get_cpu_time_sec();
    double t_start = get_time_sec();

    size_t total_completed = 0;

    // Process in chunks of MAX_BATCH_SIZE (4096)
    for (size_t offset_idx = 0; offset_idx < num_reads; offset_idx += MAX_BATCH_SIZE) {
        unsigned int current_batch_size = (num_reads - offset_idx > MAX_BATCH_SIZE) 
                                          ? MAX_BATCH_SIZE 
                                          : (unsigned int)(num_reads - offset_idx);

        CUFILE_CHECK(cuFileBatchIOSubmit(batch_handle, current_batch_size, &io_params[offset_idx], 0));

        unsigned int completed = current_batch_size;
        CUFILE_CHECK(cuFileBatchIOGetStatus(batch_handle, current_batch_size, &completed, events, NULL));
        total_completed += completed;
    }

    double t_end = get_time_sec();
    double cpu_end = get_cpu_time_sec();

    print_stats("GDS TEST 3: Full File Sequential 4KB Reads (via cuFile Batch Async)", 
                total_completed * SECTOR_4K, num_reads, t_end - t_start, cpu_end - cpu_start);

    cuFileBatchIODestroy(batch_handle);
    free(io_params);
    CUFILE_CHECK(cuFileBufDeregister(d_buffer));
    CUDA_CHECK(cudaFree(d_buffer));
    cuFileHandleDeregister(cf_handle);
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
        filename2 = argv[1];
        filename3 = argv[1];
    }

    CUDA_CHECK(cudaSetDevice(GPU_ID));
    CUFILE_CHECK(cuFileDriverOpen());

    printf("\nStarting GPUDirect Storage (GDS) Benchmark Suite:\n\n");

    run_test1_gds_2m_chunks(filename1);
    run_test2_gds_random_4k(filename2);
    run_test3_gds_seq_4k(filename3);

    cuFileDriverClose();
    return 0;
}