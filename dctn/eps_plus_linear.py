from functools import reduce
import operator
from typing import Tuple, Dict

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

    def root_mean_squares(self) -> Dict[str, torch.Tensor]:
        return {
            name: param.norm(p="fro") / param.nelement() ** 0.5
            for name, param in self.named_parameters()
        }

    @property
    def epses(self) -> Tuple[EPS, ...]:
        return tuple(module for module in self if isinstance(module, EPS))

    def l2_regularizer(self) -> torch.Tensor:
        """Returns sum of squared frobenius norms of epses' cores and the weight of the last (linear) layer.
        Note: doesn't do anything with the bias of the last (linear) layer."""
        return self[-1].weight.norm(p="fro") ** 2 + reduce(
            operator.add, (eps.core.norm(p="fro") ** 2 for eps in self.epses)
        )
