import torch
import torch.nn as nn

from einops.layers.torch import Rearrange

from dctn.eps import EPS

class EPSPlusLinear(nn.Sequential):
  def __init__(self, kernel_size, out_size):
    super().__init__(EPS(kernel_size, 1, 2, out_size),
                     Rearrange("b h w q -> b (h w q)"),
                     nn.Linear(out_size * (28-kernel_size+1)**2, 10, bias=True))
    self[-1].weight.data = torch.randn_like(self[-1].weight)
    self[-1].weight.data *= self[-1].in_features ** -0.5 / 4.
