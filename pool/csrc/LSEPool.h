#pragma once

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif

at::Tensor LSEPool_forward(
    const at::Tensor& input,
    const float r,
    const int pooled_height,
    const int pooled_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w) {
  if (input.type().is_cuda()) {
#ifdef WITH_CUDA
    return LSEPool_forward_cuda(input, r, pooled_height, pooled_width, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}
