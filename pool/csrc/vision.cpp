#include "pool.h"
#include "LSEPool.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("max_pool_forward", &MaxPool_forward, "MaxPool_forward");
  m.def("lse_pool_forward", &LSEPool_forward, "LSEPool_forward");
}
