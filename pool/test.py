from . import lse_pool
import torch
from torch.nn.functional import max_pool2d, avg_pool2d

if __name__ == '__main__':
    tensor = torch.FloatTensor([[1, 1, 2, 4],
                                [2, 3, 7, 8],
                                [3, 2, 6, 8],
                                [1, 2, 3, 7]])
    tensor = tensor.view(1, 1, 4, 4).cuda()
    print(f'input:\n{tensor}\n')
    print(f'MAX:\n{max_pool2d(tensor, kernel_size=2, stride=2, padding=0)}')
    print(f'AVG:\n{avg_pool2d(tensor, kernel_size=2, stride=2, padding=0)}')
    print(f'LSE:\n{lse_pool(tensor, 0.1, 2, 2, 0)}')
    print(lse_pool(tensor, 1.0, 2, 2, 0))
    print(lse_pool(tensor, 10.0, 2, 2, 0))
