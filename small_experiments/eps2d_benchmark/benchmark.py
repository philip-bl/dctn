import os
import itertools
from functools import reduce, partial
from timeit import timeit

from typing import *

import torch

from dctn.eps import eps2d
from dctn.benchmark import benchmark_torch

def create_tensors(
    batch_size: int,
    num_channels: int,
    height: int,
    width: int,
    kernel_size: int,
    in_size: int,
    out_size: int,
    dtype,
    device,
) -> None:
    input = torch.randn(
        num_channels, batch_size, height, width, in_size, dtype=dtype, device=device
    )
    core = torch.randn(
        *(in_size for _ in range(kernel_size ** 2 * num_channels)),
        out_size,
        dtype=dtype,
        device=device
    )
    return core, input


if __name__ == "__main__":
    device = torch.device("cuda")
    print(benchmark_torch(eps2d, partial(create_tensors, batch_size=512, num_channels=1, height=28, width=28, kernel_size=4, in_size=2, out_size=2), torch.float64, device, num_iterations=100))
