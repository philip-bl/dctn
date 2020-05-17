from functools import reduce
import math
import operator
from logging import getLogger
from typing import Tuple, Dict, Union

from more_itertools import intersperse
from attr import attrs, attrib

import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn

from einops import rearrange

from . import eps
from .eps import EPS
from . import epses_composition
from .utils import (
    ZeroCenteredNormalInitialization,
    ZeroCenteredUniformInitialization,
    FromFileInitialization,
    OneTensorInitialization,
)
from .align import make_windows
from .rank_one_tensor import RankOneTensorsBatch


@attrs(auto_attribs=True, frozen=True)
class UnitEmpiricalOutputStd:
    input: Tensor
    batch_size: int = attrib(default=128)


class UnitTheoreticalOutputStd:
    pass


@attrs(auto_attribs=True, frozen=True)
class ManuallyChosenInitialization:
    epses: Tuple[OneTensorInitialization]
    linear_weight: OneTensorInitialization
    linear_bias: OneTensorInitialization


Initialization = Union[
    UnitEmpiricalOutputStd, UnitTheoreticalOutputStd, ManuallyChosenInitialization
]


class EPSesPlusLinear(nn.Module):
    def __init__(
        self,
        epses_specs: Tuple[Tuple[int, int]],
        initialization: Initialization,
        p: float,
        device: torch.device,
        dtype: torch.dtype,
        image_size: int = 28,
    ):
        """`p` is the probability of not dropping a tensor's component."""
        assert 0.0 < p <= 1
        super().__init__()
        if isinstance(initialization, UnitEmpiricalOutputStd):
            assert initialization.input.shape[2] == image_size
            assert initialization.input.shape[3] == image_size
            epses = epses_composition.make_epses_composition_unit_empirical_output_std(
                epses_specs, initialization.input, device, dtype, initialization.batch_size
            )

        elif isinstance(initialization, UnitTheoreticalOutputStd):
            epses = epses_composition.make_epses_composition_unit_theoretical_output_std(
                epses_specs, 2, device, dtype
            )
        elif isinstance(initialization, ManuallyChosenInitialization):
            epses = tuple(
                epses_composition.make_epses_composition_manually_chosen_inializations(
                    epses_specs, initialization.epses, 2, device, dtype
                )
            )
        else:
            raise ValueError(f"{initialization=} is not {Initialization}")
        self.epses = nn.ParameterList(nn.Parameter(eps_core) for eps_core in epses)

        # initialize the linear layer
        pre_linear_image_height = (
            image_size
            - sum(kernel_sizes := tuple(ks for ks, _ in epses_specs))
            + len(kernel_sizes)
        )
        pre_linear_image_width = pre_linear_image_height
        self.linear = nn.Linear(
            pre_linear_image_height
            * pre_linear_image_width
            * eps.matrix_shape(self.epses[-1])[0],
            10,
            bias=True,
        ).to(dtype)
        if isinstance(initialization, ManuallyChosenInitialization):
            for param, param_initialization in zip(
                (self.linear.weight, self.linear.bias),
                (initialization.linear_weight, initialization.linear_bias),
            ):
                if isinstance(param_initialization, ZeroCenteredNormalInitialization):
                    new_data = torch.randn_like(param) * param_initialization.std
                elif isinstance(param_initialization, ZeroCenteredUniformInitialization):
                    new_data = (
                        torch.rand_like(param) * (2 * param_initialization.maximum)
                        - param_initialization.maximum
                    )
                else:
                    raise ValueError(
                        f"{initialization=} must be {ManuallyChosenInitialization}"
                    )
                param.data.copy_(new_data)

        else:
            self.linear.weight.data.copy_(
                torch.randn_like(self.linear.weight) * self.linear.in_features ** -0.5 / 4.0
            )
        self.linear.to(device)

        self.register_buffer("p", torch.tensor(p, device=device, dtype=dtype))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.p < 1.0 and self.training:
            epses = tuple(
                self.p.expand_as(eps_core).bernoulli() * eps_core / self.p
                for eps_core in self.epses
            )
        else:
            epses = self.epses
        intermediate = epses_composition.contract_with_input(epses, input)
        return self.linear(rearrange(intermediate, "b h w q -> b (h w q)"))

    def epswise_l2_regularizer(self) -> torch.Tensor:
        """Returns sum of squared frobenius norms of epses' cores and the weight of the last (linear) layer.
        Note: doesn't do anything with the bias of the last (linear) layer."""
        return self.linear.weight.norm(
            p="fro"
        ) ** 2 + epses_composition.epswise_squared_fro_norm(self.epses)

    def epses_composition_l2_regularizer(self) -> torch.Tensor:
        return self.linear.weight.norm(p="fro") ** 2 + epses_composition.inner_product(
            self.epses, self.epses
        )

    @torch.no_grad()
    def log_intermediate_reps_stats(self, x: torch.Tensor, batch_size: int = 128) -> None:
        logger = getLogger(f"{__name__}.{self.log_intermediate_reps_stats.__qualname__}")
        logger.info("Logging intermediate reps stats as if self.training == False")

        def log_one_tensor_stats(tensor: torch.Tensor, tensor_name: str) -> None:
            μ = tensor.mean()
            σ = tensor.std(unbiased=False)
            logger.info(
                f"{tensor_name}: {μ=:.7e}, {σ=:.7e}, {μ**2+σ**2=:.7e}, shape={tuple(tensor.shape)}"
            )

        def log_windows_stats(windows: RankOneTensorsBatch, tensor_name: str) -> None:
            μ = windows.mean_over_batch()
            σ = windows.std_over_batch(unbiased=False)
            logger.info(
                f"{tensor_name}: {μ=:.7e}, {σ=:.7e}, {μ**2+σ**2=:.7e}, "
                f"batch_shape={windows.batch_shape}, "
                f"num_factors={windows.array.shape[windows.factors_dim]}, "
                f"num_coordinates_in_one_factor={windows.array.shape[windows.coordinates_dim]}"
            )

        for n, eps_core in enumerate(self.epses):
            # log stats of the intermediate representations before contracting eps_core
            log_one_tensor_stats(x, f"x_{n}")
            kernel_size = math.isqrt(eps_core.ndim - 1)
            assert kernel_size ** 2 == eps_core.ndim - 1
            w: RankOneTensorsBatch = make_windows(x, kernel_size)
            log_windows_stats(w, f"w_{n}")
            x = eps.transform_in_slices(eps_core, x, batch_size)

        x = rearrange(x, "() b h w q -> b (h w q)")
        log_one_tensor_stats(x, f"x_{len(self.epses)}")
        output_of_linear_without_bias = F.linear(x, self.linear.weight)
        log_one_tensor_stats(output_of_linear_without_bias, "output_of_linear_without_bias")
        log_one_tensor_stats(self.linear(x), "output_of_linear_with_bias")
