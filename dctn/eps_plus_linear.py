from functools import reduce
import operator
from typing import Tuple, Dict, Union

from more_itertools import intersperse
from attr import attrs, attrib

import torch
from torch import Tensor
import torch.nn as nn

from einops.layers.torch import Rearrange

from . import eps
from dctn.eps import EPS
from . import epses_composition


@attrs(auto_attribs=True, frozen=True)
class UnitEmpiricalOutputStd:
    input: Tensor
    batch_size = attrib(default=128)


class UnitTheoreticalOutputStd:
    pass


Initialization = Union[UnitEmpiricalOutputStd, UnitTheoreticalOutputStd]


class EPSesPlusLinear(nn.Module):
    def __init__(
        self,
        epses_specs: Tuple[Tuple[int, int]],
        initialization: Initialization,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        if isinstance(initialization, UnitEmpiricalOutputStd):
            self.epses = nn.ParameterList(
                epses_composition.make_epses_composition_unit_empirical_output_std(
                    epses_specs, initialization.input, device, dtype, initialization.batch_size
                )
            )
        elif isinstance(initialization, UnitTheoreticalOutputStd):
            self.epses = nn.ParameterList(
                epses_composition.make_epses_composition_unit_theoretical_output_std(
                    epses_specs, 2, device, dtype
                )
            )
        else:
            raise ValueError(f"{initialization=} is not {Initialization}")
        pre_linear_image_height = (
            28 - (kernel_sizes := tuple(ks for ks, _ in epses_specs)) + len(kernel_sizes)
        )
        pre_linear_image_width = pre_linear_image_height
        self.linear = nn.Linear(
            pre_linear_image_height
            * pre_linear_image_width
            * eps.matrix_shape(self.epses[-1])[0],
            10,
            bias=True,
        ).to(dtype)
        self.linear.weight.data.copy_(
            torch.randn_like(self.linear.weight) * self.linear.in_features ** -0.5 / 4.0
        )
        # TODO where's dropout??!

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()  # Stopped here, TODO implement this and also see the whiteboard


class EPSesPlusLinear(nn.Sequential):
    def __init__(self, epses_specs: Tuple[Tuple[int, int]]):
        self.epses_specs = epses_specs
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

    def init_epses_composition_unit_empirical_output_std(
        self, input: torch.Tensor, batch_size: int = 128
    ) -> None:
        device = self.epses[0].core.device
        dtype = self.epses[0].core.dtype
        better_epses: Tuple[
            torch.Tensor, ...
        ] = epses_composition.make_epses_composition_unit_empirical_output_std(
            self.epses_specs, input, device, dtype
        )
        for eps_module, better_eps in zip(self.epses, better_epses):
            eps_module.core.data.copy_(better_eps)

    def root_mean_squares(self) -> Dict[str, torch.Tensor]:
        return {
            name: param.norm(p="fro") / param.nelement() ** 0.5
            for name, param in self.named_parameters()
        }

    @property
    def epses(self) -> Tuple[EPS, ...]:
        return tuple(module for module in self if isinstance(module, EPS))

    def epswise_l2_regularizer(self) -> torch.Tensor:
        """Returns sum of squared frobenius norms of epses' cores and the weight of the last (linear) layer.
        Note: doesn't do anything with the bias of the last (linear) layer."""
        return self[-1].weight.norm(p="fro") ** 2 + reduce(
            operator.add, (eps.core.norm(p="fro") ** 2 for eps in self.epses)
        )

    def epses_composition_fro_norm_squared(self) -> torch.Tensor:
        epses: Tuple[torch.Tensor, ...] = tuple(eps_module.core for eps_module in self.epses)
        return epses_composition.inner_product(epses, epses)

    def epses_composition_l2_regularizer(self) -> torch.Tensor:
        return self[-1].weight.norm(p="fro") ** 2 + self.epses_composition_fro_norm_squared()
