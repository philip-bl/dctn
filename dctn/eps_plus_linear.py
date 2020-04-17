from typing import Tuple

from more_itertools import intersperse

import torch
import torch.nn as nn

from einops.layers.torch import Rearrange

from dctn.eps import EPS


class EPSesPlusLinear(nn.Sequential):
    def __init__(self, epses_specs: Tuple[Tuple[int, int]]):
        kernel_sizes = tuple(kernel_size for kernel_size, _ in epses_specs)
        out_sizes = tuple(out_size for _, out_size in epses_specs)
        in_sizes = (2,) + out_sizes[:-1]
        epses = tuple(EPS(k, 1, i, o) for k, i, o in zip(kernel_sizes, in_sizes, out_sizes))
        linear = nn.Linear(
            (28 - sum(kernel_sizes) + len(kernel_sizes)) ** 2 * out_sizes[-1], 10, bias=True
        )
        linear.weight.data = torch.randn_like(linear.weight)
        linear.weight.data *= linear.in_features ** -0.5 / 4.0
        unsqueezer = Rearrange("b h w q -> () b h w q")
        super().__init__(
            *intersperse(unsqueezer, epses), Rearrange("b h w q -> b (h w q)"), linear
        )
