import os
import itertools
from functools import reduce, partial

from typing import *

import torch

from dctn.eps import eps


torch.tensor(1.0, device="cuda")

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
  return (
    torch.randn(
      *(in_size for _ in range(kernel_size ** 2 * num_channels)),
      out_size,
      dtype=dtype,
      device=device,
      requires_grad=True
    ),
    torch.randn(
      num_channels,
      batch_size,
      height,
      width,
      in_size,
      dtype=dtype,
      device=device,
      requires_grad=True,
    ),
  )

core, input = create_tensors(512, 1, 28, 28, 4, 2, 2, torch.float64, "cuda")

# warm up
result = eps(core, input)
ones = torch.ones_like(result)
result.backward(ones)

# actually do the thing
for i in range(30):
  result = eps(core, input)
  result.backward(ones)
