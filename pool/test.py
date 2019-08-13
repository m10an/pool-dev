from . import max_pool
import torch
from torch.nn.functional import max_pool2d

if __name__ == '__main__':
    tensor = torch.zeros(1, 1, 10, 10).cuda()
    tensor[:, :, 2, 2] = 1
    tensor[:, :, 2, 3] = 2
    tensor[:, :, 7, 3] = 3
    print(tensor)
    print(max_pool2d(tensor, kernel_size=3, stride=2, padding=1))
    print(max_pool(tensor, 3, 2, 1))
