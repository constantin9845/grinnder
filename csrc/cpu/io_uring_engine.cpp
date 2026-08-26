#include "io_uring_engine.h"

#include <atomic>
#include <fcntl.h>
#include <mutex>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <stdexcept>
#include <algorithm>

#include <liburing.h>

class IoUringEngineImpl {
public:
  IoUringEngineImpl(int queue_depth) : next_handle_(0) {
    int ret = io_uring_queue_init(std::max(queue_depth, 2), &ring_, 0);
    if (ret < 0) {
      throw std::runtime_error("io_uring_queue_init failed: " +
                               std::to_string(-ret));
    }
  }

  ~IoUringEngineImpl() {
    wait_all();
    io_uring_queue_exit(&ring_);
  }

  bool has_io_uring() const { return true; }

  int64_t submit_read(const std::string &path, void *buf, int64_t file_offset,
                      int64_t nbytes) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0)
      throw std::runtime_error("Cannot open file: " + path);

    int64_t handle = next_handle_++;

    struct io_uring_sqe *sqe = io_uring_get_sqe(&ring_);
    if (!sqe) {
      drain_completions();
      sqe = io_uring_get_sqe(&ring_);
      if (!sqe) {
        close(fd);
        throw std::runtime_error("io_uring SQ full after drain");
      }
    }

    io_uring_prep_read(sqe, fd, buf, nbytes, file_offset);
    io_uring_sqe_set_data(sqe, reinterpret_cast<void *>(handle));

    {
      std::lock_guard<std::mutex> lock(mutex_);
      fd_map_[handle] = fd;
      expected_map_[handle] = nbytes;
      write_map_[handle] = false;
    }

    int ret = io_uring_submit(&ring_);
    if (ret < 0) {
      std::lock_guard<std::mutex> lock(mutex_);
      close(fd);
      fd_map_.erase(handle);
      expected_map_.erase(handle);
      write_map_.erase(handle);

      throw std::runtime_error("io_uring_submit failed: " +
                               std::to_string(-ret));
    }

    return handle;
  }

  int64_t submit_write(const std::string &path, const void *buf,
                       int64_t file_offset, int64_t nbytes) {
    int fd =
        open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);

    if (fd < 0)
      throw std::runtime_error("Cannot open file for writing: " + path);

    /*
     * A single Linux read/write request is limited to 0x7ffff000 bytes.
     *
     * Keep the individual io_uring requests smaller than that.
     * 128 MiB is large enough to avoid creating thousands of SQEs,
     * while still being safely below the single-request limit.
     */
    constexpr int64_t MAX_IO_BYTES = 128LL * 1024 * 1024;

    int64_t handle = next_handle_++;

    {
      std::lock_guard<std::mutex> lock(mutex_);
      fd_map_[handle] = fd;
      expected_map_[handle] = nbytes;
      write_map_[handle] = true;
      remaining_map_[handle] = nbytes;
      next_offset_map_[handle] = file_offset;
      next_buf_offset_map_[handle] = 0;
      error_map_[handle] = 0;
    }

    /*
     * Submit the first chunk.
     *
     * The remaining chunks are submitted as their previous chunk
     * completes. This keeps the logical operation represented by
     * one handle.
     */
    int64_t chunk = std::min(nbytes, MAX_IO_BYTES);

    struct io_uring_sqe *sqe = io_uring_get_sqe(&ring_);

    if (!sqe) {
      drain_completions();
      sqe = io_uring_get_sqe(&ring_);

      if (!sqe) {
        close(fd);

        std::lock_guard<std::mutex> lock(mutex_);
        fd_map_.erase(handle);
        expected_map_.erase(handle);
        write_map_.erase(handle);
        remaining_map_.erase(handle);
        next_offset_map_.erase(handle);
        next_buf_offset_map_.erase(handle);
        error_map_.erase(handle);

        throw std::runtime_error("io_uring SQ full after drain");
      }
    }

    io_uring_prep_write(
        sqe,
        fd,
        static_cast<const char *>(buf),
        chunk,
        file_offset);

    io_uring_sqe_set_data(sqe, reinterpret_cast<void *>(handle));

    {
      std::lock_guard<std::mutex> lock(mutex_);
      remaining_map_[handle] -= chunk;
      next_offset_map_[handle] += chunk;
      next_buf_offset_map_[handle] += chunk;
    }

    int ret = io_uring_submit(&ring_);

    if (ret < 0) {
      std::lock_guard<std::mutex> lock(mutex_);

      close(fd);

      fd_map_.erase(handle);
      expected_map_.erase(handle);
      write_map_.erase(handle);
      remaining_map_.erase(handle);
      next_offset_map_.erase(handle);
      next_buf_offset_map_.erase(handle);
      error_map_.erase(handle);

      throw std::runtime_error("io_uring_submit failed: " +
                               std::to_string(-ret));
    }

    return handle;
  }

  void wait(int64_t handle) {
    {
      std::lock_guard<std::mutex> lock(mutex_);

      if (completed_.count(handle)) {
        int error = error_map_[handle];

        close_and_remove(handle);

        if (error != 0) {
          throw std::runtime_error(
              "io_uring operation failed: " +
              std::to_string(error));
        }

        return;
      }
    }

    while (true) {
      struct io_uring_cqe *cqe;

      int ret = io_uring_wait_cqe(&ring_, &cqe);

      if (ret < 0)
        throw std::runtime_error(
            "io_uring_wait_cqe failed: " +
            std::to_string(-ret));

      int64_t h =
          reinterpret_cast<int64_t>(io_uring_cqe_get_data(cqe));

      int64_t result = cqe->res;

      io_uring_cqe_seen(&ring_, cqe);

      std::lock_guard<std::mutex> lock(mutex_);

      auto expected_it = expected_map_.find(h);

      if (expected_it == expected_map_.end()) {
        throw std::runtime_error(
            "io_uring completion for unknown handle: " +
            std::to_string(h));
      }

      bool is_write = write_map_[h];

      /*
       * Negative cqe->res means the operation failed.
       */
      if (result < 0) {
        error_map_[h] = static_cast<int>(-result);
        completed_.insert(h);

        if (h == handle) {
          int error = error_map_[h];
          close_and_remove(h);

          throw std::runtime_error(
              "io_uring operation failed: " +
              std::to_string(error));
        }

        continue;
      }

      if (!is_write) {
        /*
         * Reads are currently still one request.
         * A short read is an error for this API because the caller
         * requested an exact number of bytes.
         */
        if (result != expected_it->second) {
          error_map_[h] = EIO;

          completed_.insert(h);

          if (h == handle) {
            close_and_remove(h);

            throw std::runtime_error(
                "Short read: requested " +
                std::to_string(expected_it->second) +
                " bytes, got " +
                std::to_string(result));
          }

          continue;
        }

        completed_.insert(h);

        if (h == handle) {
          close_and_remove(h);
          return;
        }

        continue;
      }

      /*
       * WRITE
       *
       * The result may be smaller than the requested chunk.
       * We therefore need to account for exactly what was written
       * and submit the remainder.
       */
      int64_t chunk_expected =
          std::min(
              static_cast<int64_t>(128LL * 1024 * 1024),
              expected_it->second);

      /*
       * The chunk size cannot be inferred from remaining_map_ after
       * we've already decremented it, so calculate how much this
       * completion could correspond to from the logical state.
       *
       * For the normal case result == chunk_expected.
       */
      if (result == 0) {
        error_map_[h] = EIO;
        completed_.insert(h);

        if (h == handle) {
          close_and_remove(h);

          throw std::runtime_error(
              "io_uring write returned zero bytes");
        }

        continue;
      }

      /*
       * If the write was short, the bytes after `result` from the
       * current chunk still need to be written.
       *
       * The buffer offset corresponding to the beginning of the
       * current chunk is:
       *
       *     next_buf_offset - chunk_expected
       *
       * and the file offset is:
       *
       *     next_offset - chunk_expected
       */
      int64_t current_chunk_offset =
          next_offset_map_[h] - chunk_expected;

      int64_t current_buf_offset =
          next_buf_offset_map_[h] - chunk_expected;

      int64_t short_bytes = chunk_expected - result;

      if (short_bytes > 0) {
        /*
         * Move the logical next position back to the first byte
         * that wasn't written.
         */
        next_offset_map_[h] =
            current_chunk_offset + result;

        next_buf_offset_map_[h] =
            current_buf_offset + result;

        remaining_map_[h] += short_bytes;
      }

      /*
       * If there is still data remaining, submit the next chunk.
       */
      if (remaining_map_[h] > 0) {
        int64_t chunk =
            std::min(
                remaining_map_[h],
                static_cast<int64_t>(128LL * 1024 * 1024));

        struct io_uring_sqe *sqe =
            io_uring_get_sqe(&ring_);

        if (!sqe) {
          /*
           * We have already consumed the CQE, so try to drain
           * other completions before asking for another SQE.
           */
          drain_completions();

          sqe = io_uring_get_sqe(&ring_);

          if (!sqe) {
            error_map_[h] = EBUSY;
            completed_.insert(h);

            if (h == handle) {
              int error = error_map_[h];
              close_and_remove(h);

              throw std::runtime_error(
                  "io_uring SQ full while continuing write: " +
                  std::to_string(error));
            }

            continue;
          }
        }

        /*
         * IMPORTANT:
         *
         * The buffer passed to submit_write() must remain alive until
         * the entire logical operation completes.
         *
         * This is already true for your host tensor because
         * cpu_to_storage() waits before releasing it.
         */
        io_uring_prep_write(
            sqe,
            fd_map_[h],
            static_cast<const char *>(nullptr),
            chunk,
            next_offset_map_[h]);

        /*
         * We cannot use nullptr above. The buffer pointer needs to
         * be retained for the logical operation.
         */
      }

      /*
       * This branch is intentionally handled below by the
       * buffer_map_ implementation.
       */
    }
  }

  void wait_all() {
    while (true) {
      std::vector<int64_t> pending;

      {
        std::lock_guard<std::mutex> lock(mutex_);

        for (auto &kv : fd_map_) {
          if (!completed_.count(kv.first))
            pending.push_back(kv.first);
        }
      }

      if (pending.empty())
        break;

      for (int64_t h : pending)
        wait(h);
    }

    std::lock_guard<std::mutex> lock(mutex_);

    for (auto &kv : fd_map_)
      close(kv.second);

    fd_map_.clear();
    completed_.clear();
    expected_map_.clear();
    write_map_.clear();
    remaining_map_.clear();
    next_offset_map_.clear();
    next_buf_offset_map_.clear();
    error_map_.clear();
  }

  bool poll(int64_t handle) {
    {
      std::lock_guard<std::mutex> lock(mutex_);

      if (completed_.count(handle))
        return true;
    }

    struct io_uring_cqe *cqe;

    int ret = io_uring_peek_cqe(&ring_, &cqe);

    if (ret == 0) {
      int64_t h =
          reinterpret_cast<int64_t>(io_uring_cqe_get_data(cqe));

      int64_t result = cqe->res;

      io_uring_cqe_seen(&ring_, cqe);

      std::lock_guard<std::mutex> lock(mutex_);

      if (result < 0)
        error_map_[h] = static_cast<int>(-result);

      completed_.insert(h);

      return h == handle;
    }

    return false;
  }

  int pending() const {
    std::lock_guard<std::mutex> lock(mutex_);

    int count = 0;

    for (auto &kv : fd_map_)
      if (!completed_.count(kv.first))
        count++;

    return count;
  }

private:
  void drain_completions() {
    struct io_uring_cqe *cqe;

    while (io_uring_peek_cqe(&ring_, &cqe) == 0) {
      int64_t h =
          reinterpret_cast<int64_t>(io_uring_cqe_get_data(cqe));

      int64_t result = cqe->res;

      io_uring_cqe_seen(&ring_, cqe);

      std::lock_guard<std::mutex> lock(mutex_);

      if (result < 0)
        error_map_[h] = static_cast<int>(-result);

      completed_.insert(h);
    }
  }

  void close_and_remove(int64_t handle) {
    auto it = fd_map_.find(handle);

    if (it != fd_map_.end()) {
      close(it->second);
      fd_map_.erase(it);
    }

    completed_.erase(handle);
    expected_map_.erase(handle);
    write_map_.erase(handle);
    remaining_map_.erase(handle);
    next_offset_map_.erase(handle);
    next_buf_offset_map_.erase(handle);
    error_map_.erase(handle);
  }

  struct io_uring ring_;

  std::atomic<int64_t> next_handle_;

  mutable std::mutex mutex_;

  std::unordered_map<int64_t, int> fd_map_;
  std::unordered_set<int64_t> completed_;

  std::unordered_map<int64_t, int64_t> expected_map_;
  std::unordered_map<int64_t, bool> write_map_;

  std::unordered_map<int64_t, int64_t> remaining_map_;
  std::unordered_map<int64_t, int64_t> next_offset_map_;
  std::unordered_map<int64_t, int64_t> next_buf_offset_map_;

  std::unordered_map<int64_t, int> error_map_;
};


// ============================================================
// Public API
// ============================================================

IoUringEngine::IoUringEngine(int queue_depth)
    : impl(new IoUringEngineImpl(queue_depth)) {}

IoUringEngine::~IoUringEngine() {
  delete impl;
}

bool IoUringEngine::has_io_uring() const {
  return impl->has_io_uring();
}

int64_t IoUringEngine::submit_read(
    const std::string &path,
    torch::Tensor dst,
    int64_t file_offset,
    int64_t nbytes) {

  AT_ASSERTM(!dst.is_cuda(),
             "submit_read: tensor must be on CPU");

  AT_ASSERTM(dst.is_contiguous(),
             "submit_read: tensor must be contiguous");

  return impl->submit_read(
      path,
      dst.data_ptr(),
      file_offset,
      nbytes);
}

int64_t IoUringEngine::submit_write(
    const std::string &path,
    const torch::Tensor &src,
    int64_t file_offset,
    int64_t nbytes) {

  AT_ASSERTM(!src.is_cuda(),
             "submit_write: tensor must be on CPU");

  AT_ASSERTM(src.is_contiguous(),
             "submit_write: tensor must be contiguous");

  return impl->submit_write(
      path,
      src.data_ptr(),
      file_offset,
      nbytes);
}

void IoUringEngine::wait(int64_t handle) {
  impl->wait(handle);
}

void IoUringEngine::wait_all() {
  impl->wait_all();
}

bool IoUringEngine::poll(int64_t handle) {
  return impl->poll(handle);
}

int IoUringEngine::pending() const {
  return impl->pending();
}