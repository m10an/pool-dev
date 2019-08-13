from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from ._C import max_pool_forward


def _to_h_w(param):
    return param, param if isinstance(param, int) else param


class _MaxPool(Function):
    @staticmethod
    def forward(ctx, input, kernel_size, stride, padding):
        kernel_h, kernel_w = _to_h_w(kernel_size)
        stride_h, stride_w = _to_h_w(stride)
        pad_h, pad_w = _to_h_w(padding)
        pooled_height = int(1 + (input.shape[2] + 2*pad_h - kernel_h) / stride_h)
        pooled_width = int(1 + (input.shape[3] + 2*pad_w - kernel_w) / stride_w)
        output = max_pool_forward(
            input, pooled_height, pooled_width, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w
        )
        return output

max_pool = _MaxPool.apply

# class MaxPool(nn.Module):
#     def __init__(self, kernel_size, stride, padding):
#         super(MaxPool, self).__init__()
#         self.output_size = output_size
#         self.spatial_scale = spatial_scale
#         self.output_size = output_size
#         self.spatial_scale = spatial_scale
#
#     def forward(self, input, rois):
#         return _ROIPool.apply(input, rois, self.output_size, self.spatial_scale)
#
#     def __repr__(self):
#         return self.__class__.__name__ + f'(output_size={self.output_size})'
