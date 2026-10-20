#include <torch/extension.h>

#include "cpu/subgraph.h"
#include "cpu/io_uring_engine.h"
#include "cuda/async_transfer.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Async GPU <-> Host transfers
  
  m.def("test_ugds_link", &test_ugds_link);
}
