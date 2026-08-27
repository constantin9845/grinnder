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
#include <algorithm>
#include <stdexcept>

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

    // 1GB maximum chunk size to stay safely below 2GB (INT_MAX / MAX_RW_COUNT) kernel caps
    constexpr int64_t MAX_CHUNK = 1024 * 1024 * 1024;
    uint8_t *ptr = static_cast<uint8_t *>(buf);
    int64_t bytes_remaining = nbytes;
    int64_t curr_offset = file_offset;
    int sub_ops_count = 0;

    while (bytes_remaining > 0) {
      int64_t chunk_size = std::min(bytes_remaining, MAX_CHUNK);

      struct io_uring_sqe *sqe = io_uring_get_sqe(&ring_);
      if (!sqe) {
        drain_completions();
        sqe = io_uring_get_sqe(&ring_);
        if (!sqe) {
          close(fd);
          throw std::runtime_error("io_uring SQ full after drain");
        }
      }

      io_uring_prep_read(sqe, fd, ptr, chunk_size, curr_offset);
      // Encode handle in user data
      io_uring_sqe_set_data(sqe, reinterpret_cast<void *>(handle));

      bytes_remaining -= chunk_size;
      ptr += chunk_size;
      curr_offset += chunk_size;
      sub_ops_count++;
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      fd_map_[handle] = fd;
      pending_sub_ops_[handle] = sub_ops_count;
    }

    int ret = io_uring_submit(&ring_);
    if (ret < 0) {
      std::lock_guard<std::mutex> lock(mutex_);
      close(fd);
      fd_map_.erase(handle);
      pending_sub_ops_.erase(handle);
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

    int64_t handle = next_handle_++;

    // 1GB maximum chunk size to prevent u32 / MAX_RW_COUNT overflow in kernel
    constexpr int64_t MAX_CHUNK = 1024 * 1024 * 1024;
    const uint8_t *ptr = static_cast<const uint8_t *>(buf);
    int64_t bytes_remaining = nbytes;
    int64_t curr_offset = file_offset;
    int sub_ops_count = 0;

    while (bytes_remaining > 0) {
      int64_t chunk_size = std::min(bytes_remaining, MAX_CHUNK);

      struct io_uring_sqe *sqe = io_uring_get_sqe(&ring_);
      if (!sqe) {
        drain_completions();
        sqe = io_uring_get_sqe(&ring_);
        if (!sqe) {
          close(fd);
          throw std::runtime_error("io_uring SQ full after drain");
        }
      }

      io_uring_prep_write(sqe, fd, ptr, chunk_size, curr_offset);
      io_uring_sqe_set_data(sqe, reinterpret_cast<void *>(handle));

      bytes_remaining -= chunk_size;
      ptr += chunk_size;
      curr_offset += chunk_size;
      sub_ops_count++;
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      fd_map_[handle] = fd;
      pending_sub_ops_[handle] = sub_ops_count;
    }

    int ret = io_uring_submit(&ring_);
    if (ret < 0) {
      std::lock_guard<std::mutex> lock(mutex_);
      close(fd);
      fd_map_.erase(handle);
      pending_sub_ops_.erase(handle);
      throw std::runtime_error("io_uring_submit failed: " +
                               std::to_string(-ret));
    }
    return handle;
  }

  void wait(int64_t handle) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (completed_.count(handle)) {
        close_and_remove(handle);
        return;
      }
    }

    while (true) {
      struct io_uring_cqe *cqe;
      int ret = io_uring_wait_cqe(&ring_, &cqe);
      if (ret < 0)
        throw std::runtime_error("io_uring_wait_cqe failed: " +
                                 std::to_string(-ret));

      int64_t h = reinterpret_cast<int64_t>(io_uring_cqe_get_data(cqe));
      int res = cqe->res;
      io_uring_cqe_seen(&ring_, cqe);

      if (res < 0) {
        throw std::runtime_error("io_uring operation failed with error code: " +
                                 std::to_string(-res));
      }

      std::lock_guard<std::mutex> lock(mutex_);
      if (--pending_sub_ops_[h] == 0) {
        completed_.insert(h);
        pending_sub_ops_.erase(h);
      }

      if (h == handle && completed_.count(handle)) {
        close_and_remove(h);
        return;
      }
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
    pending_sub_ops_.clear();
  }

  bool poll(int64_t handle) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (completed_.count(handle))
        return true;
    }

    drain_completions();

    std::lock_guard<std::mutex> lock(mutex_);
    return completed_.count(handle) > 0;
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
      int64_t h = reinterpret_cast<int64_t>(io_uring_cqe_get_data(cqe));
      int res = cqe->res;
      io_uring_cqe_seen(&ring_, cqe);

      if (res < 0) {
        throw std::runtime_error("io_uring operation failed with error code: " +
                                 std::to_string(-res));
      }

      std::lock_guard<std::mutex> lock(mutex_);
      if (--pending_sub_ops_[h] == 0) {
        completed_.insert(h);
        pending_sub_ops_.erase(h);
      }
    }
  }

  void close_and_remove(int64_t handle) {
    auto it = fd_map_.find(handle);
    if (it != fd_map_.end()) {
      close(it->second);
      fd_map_.erase(it);
    }
    completed_.erase(handle);
    pending_sub_ops_.erase(handle);
  }

  struct io_uring ring_;
  std::atomic<int64_t> next_handle_;
  mutable std::mutex mutex_;
  std::unordered_map<int64_t, int> fd_map_;
  std::unordered_map<int64_t, int> pending_sub_ops_;
  std::unordered_set<int64_t> completed_;
};