#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>


#define CUDA_1D_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)


template <typename Dtype>
__global__ void MaxPoolFForward(
    const int nthreads,
    const Dtype* const bottom_data,
    const int num,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    Dtype* const top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    float maxval = -FLT_MAX;
    int maxidx = -1;
    const Dtype* const bottom_slice = bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (static_cast<float>(bottom_slice[h * width + w]) > maxval) {
        	maxidx = h * width + w;
          maxval = static_cast<float>(bottom_slice[maxidx]);
        }
      }
    }
    top_data[index] = static_cast<Dtype>(maxval);
  }
}

at::Tensor MaxPool_forward_cuda(
    const at::Tensor& input,
    const int pooled_height,
    const int pooled_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w) {
  AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");

  auto num = input.size(0);
  auto channels = input.size(1);
  auto height = input.size(2);
  auto width = input.size(3);

  auto output = at::empty({num, channels, pooled_height, pooled_width}, input.options());
  auto output_size = num * pooled_height * pooled_width * channels;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(THCCeilDiv(output_size, 512L), 4096L));
  dim3 block(512);

  if (output.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return output;
  }

  AT_DISPATCH_FLOATING_TYPES(input.type(), "MaxPool_forward", [&] {
    MaxPoolFForward<scalar_t><<<grid, block, 0, stream>>>(
      output_size,
      input.contiguous().data<scalar_t>(),
      num,
      channels,
      height,
      width,
      pooled_height,
      pooled_width,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      output.data<scalar_t>());
  });
  THCudaCheck(cudaGetLastError());
  return output;
}
