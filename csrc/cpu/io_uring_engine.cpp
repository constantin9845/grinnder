#include "io_uring_engine.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>

#include <liburing.h>

// All file I/O goes through O_DIRECT, so it never fills the page cache: the
// only host copy of the data is the caller's buffer.
//
// O_DIRECT needs the buffer address, the length and the file offset to be
// multiples of the device's logical block size (4 KiB covers our devices).
//   - Aligned buffer: read/written in place, in kChunk requests. A partial
//     last block goes through a small aligned buffer; writes pad it with zeros
//     and then truncate the file back to its exact size.
//   - Unaligned buffer: copied through an aligned bounce buffer one chunk at
//     a time, synchronously (slower; HostBuffer allocates aligned tensors so
//     the hot path never takes it).

namespace {

constexpr int64_t kAlign = 4096;
constexpr int64_t kChunk = 8LL << 20;

inline bool is_aligned(const void *p) {
  return (reinterpret_cast<uintptr_t>(p) & (kAlign - 1)) == 0;
}

inline int64_t round_up(int64_t n) { return (n + kAlign - 1) & ~(kAlign - 1); }

void *alloc_aligned(size_t n) {
  void *p = nullptr;
  if (posix_memalign(&p, kAlign, n) != 0)
    throw std::bad_alloc();
  return p;
}

int open_direct(const std::string &path, int flags) {
  int fd = open(path.c_str(), flags | O_DIRECT, S_IRUSR | S_IWUSR);
  if (fd < 0) {
    int e = errno;
    std::string msg = "Cannot open " + path + " with O_DIRECT: " + strerror(e);
    if (e == EINVAL)
      msg += " (the filesystem does not support O_DIRECT)";
    throw std::runtime_error(msg);
  }
  return fd;
}

} // namespace

class IoUringEngineImpl {
public:
  explicit IoUringEngineImpl(int queue_depth)
      : depth_(std::max(queue_depth, 2)), next_handle_(0) {
    int ret = io_uring_queue_init(depth_, &ring_, 0);
    if (ret < 0) {
      throw std::runtime_error("io_uring_queue_init failed: " +
                               std::to_string(-ret));
    }
    bounce_ = alloc_aligned(kChunk);
  }

  ~IoUringEngineImpl() {
    try {
      wait_all();
    } catch (...) {
    }
    io_uring_queue_exit(&ring_);
    free(bounce_);
  }

  bool has_io_uring() const { return true; }

  int64_t submit_read(const std::string &path, void *buf, int64_t file_offset,
                      int64_t nbytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    check_offset(path, file_offset);
    int fd = open_direct(path, O_RDONLY);
    char *p = static_cast<char *>(buf);
    if (!is_aligned(buf))
      return bounced(path, fd, false, p, file_offset, nbytes);
    return submit_direct(fd, false, p, file_offset, nbytes);
  }

  int64_t submit_write(const std::string &path, const void *buf,
                       int64_t file_offset, int64_t nbytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    check_offset(path, file_offset);
    // A write from offset 0 replaces the file; a write at an offset adds to
    // it (chunked writes of one tensor).
    int flags = O_WRONLY | O_CREAT | (file_offset == 0 ? O_TRUNC : 0);
    int fd = open_direct(path, flags);
    char *p = const_cast<char *>(static_cast<const char *>(buf));
    if (!is_aligned(buf))
      return bounced(path, fd, true, p, file_offset, nbytes);
    return submit_direct(fd, true, p, file_offset, nbytes);
  }

  void wait(int64_t handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    while (ops_.count(handle))
      reap(true);
    take_result(handle);
  }

  void wait_all() {
    std::lock_guard<std::mutex> lock(mutex_);
    while (!ops_.empty())
      reap(true);
    int err = 0;
    for (auto &kv : finished_)
      if (kv.second != 0 && err == 0)
        err = kv.second;
    finished_.clear();
    if (err != 0)
      throw std::runtime_error(std::string("O_DIRECT I/O failed: ") +
                               strerror(-err));
  }

  bool poll(int64_t handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    while (reap(false)) {
    }
    return ops_.count(handle) == 0;
  }

  int pending() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int>(ops_.size());
  }

private:
  // One read or write call, split into one or more requests.
  struct Op {
    int fd = -1;
    bool write = false;
    int remaining = 0;   // requests not completed yet (+1 while submitting)
    int error = 0;       // first failing request's -errno
    int64_t expected = 0;
    int64_t done = 0;    // bytes transferred so far
    void *tail = nullptr;      // aligned buffer for a partial last block
    char *tail_dst = nullptr;  // where a read's tail bytes go
    int64_t tail_len = 0;
    int64_t truncate_to = -1;  // writes with a padded tail
  };

  static void check_offset(const std::string &path, int64_t file_offset) {
    if (file_offset % kAlign != 0)
      throw std::runtime_error("O_DIRECT needs a 4 KiB aligned file offset (" +
                               path + ", offset " +
                               std::to_string(file_offset) + ")");
  }

  int64_t submit_direct(int fd, bool write, char *buf, int64_t off,
                        int64_t n) {
    const int64_t handle = next_handle_++;
    Op &op = ops_[handle]; // stays valid: unordered_map nodes never move
    op.fd = fd;
    op.write = write;
    op.expected = n;
    op.remaining = 1; // keeps the op alive until every request is queued

    const int64_t bulk = n & ~(kAlign - 1);
    const int64_t tail = n - bulk;
    try {
      for (int64_t done = 0; done < bulk; done += kChunk)
        queue(handle, op, buf + done, std::min(kChunk, bulk - done),
              off + done);
      if (tail > 0) {
        op.tail = alloc_aligned(kAlign);
        op.tail_dst = buf + bulk;
        op.tail_len = tail;
        if (write) {
          memset(op.tail, 0, kAlign);
          memcpy(op.tail, buf + bulk, tail);
          op.truncate_to = off + n;
        }
        queue(handle, op, op.tail, kAlign, off + bulk);
      }
      flush();
    } catch (...) {
      // Let the requests already queued finish, then drop the op.
      if (op.error == 0)
        op.error = -EIO;
      if (--op.remaining == 0)
        finish(handle);
      throw;
    }
    if (--op.remaining == 0)
      finish(handle);
    return handle;
  }

  void queue(int64_t handle, Op &op, void *p, int64_t len, int64_t off) {
    while (inflight_ >= depth_)
      reap(true);
    io_uring_sqe *sqe = io_uring_get_sqe(&ring_);
    if (!sqe) {
      flush();
      sqe = io_uring_get_sqe(&ring_);
      if (!sqe)
        throw std::runtime_error("io_uring: no free submission entry");
    }
    if (op.write)
      io_uring_prep_write(sqe, op.fd, p, static_cast<unsigned>(len), off);
    else
      io_uring_prep_read(sqe, op.fd, p, static_cast<unsigned>(len), off);
    io_uring_sqe_set_data(sqe, reinterpret_cast<void *>(handle));
    ++op.remaining;
    ++inflight_;
  }

  void flush() {
    int ret = io_uring_submit(&ring_);
    if (ret < 0)
      throw std::runtime_error("io_uring_submit failed: " +
                               std::to_string(-ret));
  }

  // Handle one completion. blocking=false returns false if none is ready.
  bool reap(bool blocking) {
    if (inflight_ == 0)
      return false;
    flush();
    io_uring_cqe *cqe = nullptr;
    int ret = blocking ? io_uring_wait_cqe(&ring_, &cqe)
                       : io_uring_peek_cqe(&ring_, &cqe);
    if (ret == -EAGAIN || ret == -EINTR)
      return false;
    if (ret < 0)
      throw std::runtime_error("io_uring wait failed: " +
                               std::to_string(-ret));
    const int64_t h = reinterpret_cast<int64_t>(io_uring_cqe_get_data(cqe));
    const int res = cqe->res;
    io_uring_cqe_seen(&ring_, cqe);
    --inflight_;

    auto it = ops_.find(h);
    if (it != ops_.end()) {
      Op &op = it->second;
      if (res < 0) {
        if (op.error == 0)
          op.error = res;
      } else {
        op.done += res;
      }
      if (--op.remaining == 0)
        finish(h);
    }
    return true;
  }

  void finish(int64_t h) {
    auto it = ops_.find(h);
    Op &op = it->second;
    int err = op.error;
    if (err == 0 && op.done < op.expected)
      err = -EIO; // short transfer (file smaller than requested)
    if (err == 0 && !op.write && op.tail)
      memcpy(op.tail_dst, op.tail, op.tail_len);
    if (err == 0 && op.truncate_to >= 0 && ftruncate(op.fd, op.truncate_to) != 0)
      err = -errno;
    close(op.fd);
    free(op.tail);
    finished_[h] = err;
    ops_.erase(it);
  }

  void take_result(int64_t h) {
    auto it = finished_.find(h);
    if (it == finished_.end())
      return;
    int err = it->second;
    finished_.erase(it);
    if (err != 0)
      throw std::runtime_error(std::string("O_DIRECT I/O failed: ") +
                               strerror(-err));
  }

  // Unaligned buffer: copy through the bounce buffer, chunk by chunk.
  // Synchronous; returns an already finished handle.
  int64_t bounced(const std::string &path, int fd, bool write, char *buf,
                  int64_t off, int64_t n) {
    if (!warned_unaligned_) {
      warned_unaligned_ = true;
      fprintf(stderr,
              "[io_uring] buffer not 4 KiB aligned (%s): using a bounce "
              "buffer for O_DIRECT (slower)\n",
              path.c_str());
    }
    char *b = static_cast<char *>(bounce_);
    int err = 0;
    bool padded = false;
    for (int64_t done = 0; done < n && err == 0; done += kChunk) {
      const int64_t len = std::min(kChunk, n - done);
      const int64_t io_len = round_up(len);
      if (write) {
        memcpy(b, buf + done, len);
        if (io_len > len) {
          memset(b + len, 0, io_len - len);
          padded = true;
        }
        ssize_t r = pwrite(fd, b, io_len, off + done);
        if (r != io_len)
          err = r < 0 ? -errno : -EIO;
      } else {
        ssize_t r = pread(fd, b, io_len, off + done);
        if (r < len)
          err = r < 0 ? -errno : -EIO;
        else
          memcpy(buf + done, b, len);
      }
    }
    if (err == 0 && padded && ftruncate(fd, off + n) != 0)
      err = -errno;
    close(fd);
    const int64_t handle = next_handle_++;
    finished_[handle] = err;
    return handle;
  }

  struct io_uring ring_;
  const int depth_;
  int inflight_ = 0;
  std::atomic<int64_t> next_handle_;
  mutable std::mutex mutex_;
  std::unordered_map<int64_t, Op> ops_;      // in flight
  std::unordered_map<int64_t, int> finished_; // done, result not taken yet
  void *bounce_ = nullptr;
  bool warned_unaligned_ = false;
};

// ============================================================
// Public API
// ============================================================

IoUringEngine::IoUringEngine(int queue_depth)
    : impl(new IoUringEngineImpl(queue_depth)) {}

IoUringEngine::~IoUringEngine() { delete impl; }

bool IoUringEngine::has_io_uring() const { return impl->has_io_uring(); }

int64_t IoUringEngine::submit_read(const std::string &path, torch::Tensor dst,
                                   int64_t file_offset, int64_t nbytes) {
  AT_ASSERTM(!dst.is_cuda(), "submit_read: tensor must be on CPU");
  AT_ASSERTM(dst.is_contiguous(), "submit_read: tensor must be contiguous");
  return impl->submit_read(path, dst.data_ptr(), file_offset, nbytes);
}

int64_t IoUringEngine::submit_write(const std::string &path, torch::Tensor src,
                                    int64_t file_offset, int64_t nbytes) {
  AT_ASSERTM(!src.is_cuda(), "submit_write: tensor must be on CPU");
  AT_ASSERTM(src.is_contiguous(), "submit_write: tensor must be contiguous");
  return impl->submit_write(path, src.data_ptr(), file_offset, nbytes);
}

void IoUringEngine::wait(int64_t handle) { impl->wait(handle); }
void IoUringEngine::wait_all() { impl->wait_all(); }
bool IoUringEngine::poll(int64_t handle) { return impl->poll(handle); }
int IoUringEngine::pending() const { return impl->pending(); }
