from typing import Callable

import torch
from torch import Tensor

@torch.no_grad()
def transform_dataset(f: Callable[[Tensor], Tensor], x: Tensor, batch_size: int = 64):
  """`x` must be of shape channel×sample×height×width×quantum.
  `f` must be an `eps`-like function which takes one argument - input - and produces the output.
  Doesn't propagate gradient."""
  return torch.cat(tuple(f(slice) for slice in torch.split(x, batch_size, dim=1))).unsqueeze(0)
