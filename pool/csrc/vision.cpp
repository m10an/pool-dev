#include "pool.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("max_pool_forward", &MaxPool_forward, "MaxPool_forward");
}
