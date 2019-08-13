#pragma once
#include <torch/extension.h>


at::Tensor MaxPool_forward_cuda(const at::Tensor& input,
                                const int pooled_height,
                                const int pooled_width,
                                const int kernel_h,
                                const int kernel_w,
                                const int stride_h,
                                const int stride_w,
                                const int pad_h,
                                const int pad_w);
