#include "async_transfer.h"

#include <ATen/cuda/CUDAContext.h>

#include "../thread_pool.h"

#include <fcntl.h>
#include <unistd.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <cstring>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h> 
#include <cuda_runtime.h>
#include <future>
#include <cstdlib>
#include <mutex>
#include <algorithm>
#include <cstdint>
#include <cerrno>
#include <sys/ioctl.h>
#include <linux/nvme_ioctl.h>
#include <cstdio>



// python3 setup.py build_ext --inplace

// compiler errors for build : 
// export CC=/usr/bin/gcc
// export CXX=/usr/bin/g++
// GRINNDER_USE_UGDS=1 python -m pip install -e . --no-build-isolation

#include <ugds.h>

uGDSError_t test_ugds_link() {
    return uGDSDriverOpen();
}