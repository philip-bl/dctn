import itertools
import functools
import math
from typing import *

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import opt_einsum as oe

from dctn.align import align

def eps(core: Tensor, input: Tensor) -> Tensor:
  num_channels, batch_size, height, width, in_size = input.shape
  kernel_size = math.isqrt((core.ndim - 1) // num_channels)
  assert core.shape[:-1] == tuple(
    in_size for _ in range(kernel_size ** 2 * num_channels)
  )
  out_size = core.shape[-1]
  aligned_input_cores = tuple(align(input, kernel_size))
  contraction_path = (
    tuple(range(input_part_0_len := math.ceil(len(aligned_input_cores) / 2))),
    tuple(range(len(aligned_input_cores) - input_part_0_len)),
    (0, 1),
    (0, 1),
  )
  return oe.contract(
    *itertools.chain.from_iterable(
      (input_core, ("batch", "height", "width", f"in{index}"))
      for index, input_core in enumerate(aligned_input_cores)
    ),
    core,
    tuple(f"in{index}" for index in range(len(aligned_input_cores))) + ("out",),
    ("batch", "height", "width", "out"),
    optimize=contraction_path,
  )


def eps_one_by_one(core: Tensor, input: Tensor) -> Tensor:
  num_channels, batch_size, height, width, in_size = input.shape
  kernel_size = math.isqrt((core.ndim - 1) // num_channels)
  assert core.shape[:-1] == tuple(
    in_size for _ in range(kernel_size ** 2 * num_channels)
  )
  out_size = core.shape[-1]
  aligned_input_cores = align(input, kernel_size)
  first = True
  for input_core in aligned_input_cores:
    if first:
      intermediate = torch.einsum("bhwi,i...->bhw...", input_core, core)
      first = False
    else:
      intermediate = torch.einsum(
        "bhwi,bhwi...->bhw...", input_core, intermediate
      )

  assert intermediate.shape == (
    batch_size,
    height - kernel_size + 1,
    width - kernel_size + 1,
    out_size,
  )
  return intermediate


class EPS(nn.Module):
  def __init__(self, kernel_size: int, in_num_channels: int, in_size: int, out_size: int):
    super().__init__()
    self.kernel_size = kernel_size
    self.in_num_channels = in_num_channels
    self.in_size = in_size
    self.out_size = out_size
    std = self.matrix_shape[1] ** -0.5 # preserves std during forward pass
    self.core = nn.Parameter(
      torch.randn(*(in_size,)*(kernel_size**2 * in_num_channels), out_size)*std)

  @property
  def matrix_shape(self) -> Tuple[int, int]:
    return (self.out_size, self.in_size ** (self.kernel_size**2 * self.in_num_channels))

  def forward(self, input: Tensor) -> Tensor:
    return eps(self.core, input)
