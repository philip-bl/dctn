from attr import attrs, attrib
from itertools import chain
import functools
import operator
import logging
import math

from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F

import opt_einsum as oe

import einops

from .conv_sbs_spec import SBSSpecString, SBSSpecCore

# here goes types of initialization of ConvSBS


@attrs(auto_attribs=True, frozen=True)
class DumbNormalInitialization:
    std_of_elements_of_cores: float


@attrs(auto_attribs=True, frozen=True)
class KhrulkovNormalInitialization:
    std_of_elements_of_matrix: Optional[float]


class NormalPreservingOutputStdInitialization:
    pass


@attrs(auto_attribs=True, frozen=True)
class MinRandomEyeInitialization:
    base_std: float


class ConvSBS(nn.Module):
    def __init__(
        self,
        spec: SBSSpecString,
        initialization: Union[
            DumbNormalInitialization,
            KhrulkovNormalInitialization,
            NormalPreservingOutputStdInitialization,
            MinRandomEyeInitialization,
        ] = DumbNormalInitialization(0.9),
    ):
        super().__init__()
        logger = logging.getLogger(f"{__name__}.{self.__init__.__qualname__}")
        self.spec = spec
        self.cores = nn.ParameterList(
            (
                nn.Parameter(
                    (
                        initialization.std_of_elements_of_cores
                        if isinstance(initialization, DumbNormalInitialization)
                        else float("nan")
                    )
                    * torch.randn(*shape.as_tuple())
                )
                for shape in self.spec.shapes
            )
        )
        if isinstance(initialization, KhrulkovNormalInitialization):
            self.init_khrulkov_normal(initialization.std_of_elements_of_matrix)
        elif isinstance(initialization, NormalPreservingOutputStdInitialization):
            logger.info("Using normal preserving output std initialization")
            self.init_normal_preserving_output_std()
        elif isinstance(initialization, MinRandomEyeInitialization):
            self.init_min_random_eye(initialization.base_std)
        self._first_stage_einsum_exprs = None
        self._second_stage_einsum_expr = None

        self._sum_einsum_expr = oe.contract_expression(
            *chain.from_iterable(
                (shape.as_tuple(), dim_names)
                for shape, dim_names in zip(self.spec.shapes, self.spec.all_dim_names)
            ),
            (),  # the result is a scalar - we contract out all dimensions
            optimize="auto",
        )
        logger.info(f"{self._sum_einsum_expr=}")

        # generate the einsum expression for calculating squared frobenius norm
        # the code below joins all dimensions (of two copies of the tt network)
        # except for bond dimensions
        self._squared_fro_norm_einsum_expr = oe.contract_expression(
            *chain.from_iterable(
                (shape.as_tuple(), dim_names)
                for shape, dim_names in zip(
                    self.spec.shapes,
                    self.spec.get_all_dim_names_add_suffix_to_bonds("_a"),
                )
            ),
            *chain.from_iterable(
                (shape.as_tuple(), dim_names)
                for shape, dim_names in zip(
                    self.spec.shapes,
                    self.spec.get_all_dim_names_add_suffix_to_bonds("_b"),
                )
            ),
            (),  # the result is a scalar
            optimize="auto",
        )

        self._as_explicit_tensor_einsum_expr = oe.contract_expression(
            *chain.from_iterable(
                (shape.as_tuple(), dim_names)
                for shape, dim_names in zip(self.spec.shapes, self.spec.all_dim_names)
            ),
            self.spec.all_dangling_dim_names,
            optimize="auto",
        )

        logger.info(f"{self.var()**0.5=}")

    @property
    def tt_matrix_num_columns(self) -> Tuple[int, int]:
        return self.spec.in_quantum_dim_size ** (
            self.spec.in_num_channels * len(self.spec.cores)
        )

    def init_khrulkov_normal(
        self, std_of_elements_of_matrix: Optional[float] = None
    ) -> None:
        logger = logging.getLogger(
            f"{__name__}.{self.init_khrulkov_normal.__qualname__}"
        )
        if std_of_elements_of_matrix is not None:
            logger.info(f"desired {std_of_elements_of_matrix=}")
            var_of_elements_of_matrix = std_of_elements_of_matrix ** 2
        else:
            # See Tensorized Embedding Layers for Efficient Model Compression by Khrulkov
            # Section "Initialization"
            matrix_num_rows = self.spec.out_total_quantum_dim_size
            var_of_elements_of_matrix = 2 / (
                self.tt_matrix_num_columns + matrix_num_rows
            )
            logger.info(
                f"{self.tt_matrix_num_columns=}, {matrix_num_rows=}, {var_of_elements_of_matrix**0.5=}"
            )

        prod_of_ranks = functools.reduce(operator.mul, self.spec.bond_sizes)
        var_of_cores_elements = var_of_elements_of_matrix ** (
            1 / len(self.cores)
        ) / prod_of_ranks ** (1 / len(self.cores))
        logger.info(
            f"{self.spec.bond_sizes=}, {prod_of_ranks=}, {var_of_cores_elements=}"
        )
        for core in self.cores:
            torch.nn.init.normal_(core, std=math.sqrt(var_of_cores_elements))

    def init_normal_preserving_output_std(self) -> None:
        """Initializes each component of each core with i.i.d. normal distribution with zero mean and such
        variance, that if an input of this layer (i.e. a rank-one "window" tensor) has i.i.d. coordinates with
        mean μ and std σ, then each coordinate of the output of this layer will have std √(σ^2+μ^2)."""
        self.init_khrulkov_normal(self.tt_matrix_num_columns ** -0.5)

    def init_min_random_eye(self, base_std: float) -> None:
        """Initializes by min_random_eye, but adjusts so that the layer's output has the same mean as
        mean of an input's window. base_std will be multiplied by a factor depending on the input dimension
        of a core to determine std of gaussian noise added to cores."""
        assert self.spec.bond_sizes[0] == 1  # can't work with a tensor ring

        # also can't work with different bond sizes
        assert all(
            bond_size == self.spec.bond_sizes[1]
            for bond_size in self.spec.bond_sizes[1:]
        )
        bond_size = self.spec.bond_sizes[1]

        # can't work with multiple cores having output dimensions
        assert self.spec.out_total_quantum_dim_size == max(
            shape.out_quantum_dim_size for shape in self.spec.shapes
        )
        out_dim_size = self.spec.out_total_quantum_dim_size

        # first initialize all cores except for the first and the last
        total_in_dim_size = self.spec.in_quantum_dim_size ** self.spec.in_num_channels
        truncated_scaled_identity = torch.zeros(bond_size, bond_size)
        truncated_scaled_identity[
            : min(bond_size, out_dim_size), : min(bond_size, out_dim_size)
        ] = (torch.eye(min(bond_size, out_dim_size)) / total_in_dim_size)
        truncated_scaled_identity = einops.rearrange(
            truncated_scaled_identity,
            "l r -> () l r "
            + " ".join(("()" for _ in range(self.spec.in_num_channels))),
        )
        for (core, shape) in zip(self.cores[1:-1], self.spec.shapes[1:-1]):
            # shape is of type SBSSpecCore
            # core has shape out×l×r×in1×in2×...×inN
            nn.init.zeros_(core.data)
            core.data += truncated_scaled_identity.expand_as(core)
            core.data += torch.randn_like(core) * base_std / total_in_dim_size

        # second initialize the first and the last cores
        # cores[0] has shape 1×1×r×in1×in2×...×inN
        # cores[-1] has shape 1×r×1×in1×in2×...×inN
        for core in (self.cores[0], self.cores[-1]):
            nn.init.zeros_(core.data)
            core.data[0, 0, 0] = 1 / total_in_dim_size
            assert torch.allclose(core.data.sum(), torch.tensor(1.0))
            core.data += torch.randn_like(core) * base_std / total_in_dim_size

    @property
    def _second_stage_result_dimensions_names(self) -> Tuple[str, ...]:
        return (
            "batch",
            "height",
            "width",
            *(f"out_quantum_{i}" for i in range(len(self.cores))),
        )

    def gen_einsum_exprs(self, batch_size: int, height: int, width: int) -> None:
        self._first_stage_einsum_exprs = tuple(
            oe.contract_expression(
                shape.as_tuple(),
                shape.dimensions_names,  # the core
                *chain.from_iterable(
                    (
                        (batch_size, height, width, self.spec.in_quantum_dim_size),
                        ("batch", "height", "width", dim_name),
                    )
                    for dim_name in shape.dimensions_names
                    if dim_name.startswith("in_quantum_")
                ),  # the channels
                (
                    "batch",
                    "out_quantum",
                    "bond_left",
                    "bond_right",
                    "height",
                    "width",
                ),  # the result
                optimize="optimal",
            )
            for shape in self.spec.shapes
        )

        self._second_stage_einsum_expr = oe.contract_expression(
            *chain.from_iterable(
                (
                    (
                        batch_size,
                        shape.out_quantum_dim_size,
                        shape.bond_left_size,
                        shape.bond_right_size,
                        height,
                        width,
                    ),
                    (
                        "batch",
                        *parameters_core_dim_names[
                            :3
                        ],  # out_quantum_i, bond_i, bond_(i+1 or 0)
                        "height",
                        "width",
                    ),
                )
                for parameters_core_dim_names, shape in zip(
                    self.spec.all_dim_names, self.spec.shapes
                )
            ),
            self._second_stage_result_dimensions_names,
            optimize="auto",
        )

    def sum(self) -> torch.Tensor:
        """Returns the sum of all elements of the TT tensor."""
        return self._sum_einsum_expr(*self.cores, backend="torch")

    def mean(self) -> torch.Tensor:
        """Returns the mean of all elements of the TT tensor."""
        return self.sum() / float(self.spec.nelement)

    def squared_fro_norm(self) -> torch.Tensor:
        """Returns the squared Frobenius norm of the TT tensor."""
        return self._squared_fro_norm_einsum_expr(
            *self.cores, *self.cores, backend="torch"
        )

    def fro_norm(self) -> torch.Tensor:
        """Returns the Frobenius norm of the TT tensor."""
        return self.squared_fro_norm() ** 0.5

    def var(self, unbiased: bool = True) -> torch.Tensor:
        """Returns the empiric variance of the TT tensor.
        Applies Bessel's correction iff unbiased is True."""
        sum = self.sum()
        mean = sum / self.spec.nelement
        divisor = self.spec.nelement - 1 if unbiased else self.spec.nelement
        return (
            self.squared_fro_norm() / divisor
            - 2 * sum / divisor * mean
            + self.spec.nelement / divisor * mean ** 2
        )

    def as_explicit_tensor(self) -> torch.Tensor:
        """Returns the TT tensor as just one large multidimensional array.
        Dimensions will be ordered as self.spec.all_dangling_dim_names."""
        return self._as_explicit_tensor_einsum_expr(*self.cores, backend="torch")

    def forward(
        self, channels: Union[torch.Tensor, Tuple[torch.Tensor, ...]], /
    ) -> torch.Tensor:
        """If passing a tensor, the very first dimension MUST be channels."""
        if isinstance(channels, torch.Tensor):
            channels = tuple(channels)
        # now channels is a tuple of tensors, each tensor corresponding to a channel
        batch_size, height, width = channels[0].shape[:3]
        if (
            self._first_stage_einsum_exprs is None
            or self._second_stage_einsum_expr is None
        ):
            self.gen_einsum_exprs(batch_size, height, width)
        contracted_with_cores_separately = tuple(
            einsum_expr(core, *(channel for channel in channels), backend="torch")
            for i, (core, einsum_expr) in enumerate(
                zip(self.cores, self._first_stage_einsum_exprs)
            )
        )
        # for each i, contracted_with_cores_separately[i] is the result of contracting the ith
        # core with the input

        # TODO, instead of this, use eps.align_via_padding

        # now we do padding:
        padded = tuple(
            F.pad(
                intermediate,
                # padding size goes like (left, right, top, lower)
                [
                    max(
                        (self.spec.max_width_pos - self.spec.min_width_pos)
                        - (pos.w - self.spec.min_width_pos),
                        0,
                    ),
                    max(pos.w - self.spec.min_width_pos, 0),
                    max(
                        (self.spec.max_height_pos - self.spec.min_height_pos)
                        - (pos.h - self.spec.min_height_pos),
                        0,
                    ),
                    max(pos.h - self.spec.min_height_pos, 0),
                ],
                value=0.0,
            )
            for i, (intermediate, pos) in enumerate(
                zip(
                    contracted_with_cores_separately,
                    (core_spec.position for core_spec in self.spec.cores),
                )
            )
        )

        # now we do the second stage
        padded_result = einops.rearrange(
            self._second_stage_einsum_expr(*padded),
            "b h w {0} -> b h w ({0})".format(
                " ".join((f"q{i}" for i in range(len(self.cores))))
            ),
        )
        # TODO in test_conv_sbs change everything to b h w q
        # the good region is the region where padded value has no effect
        good_region_height_limits = (
            self.spec.max_height_pos - self.spec.min_height_pos,
            padded_result.shape[1]
            - (self.spec.max_height_pos - self.spec.min_height_pos),
        )
        good_region_width_limits = (
            self.spec.max_width_pos - self.spec.min_width_pos,
            padded_result.shape[2]
            - (self.spec.max_width_pos - self.spec.min_width_pos),
        )
        result = padded_result[
            :,
            good_region_height_limits[0] : good_region_height_limits[1],
            good_region_width_limits[0] : good_region_width_limits[1],
            :,
        ]
        return result

    def multiply_by_scalar(self, scalar: float, /):
        """ConvSBS is a tensor network. This method multiplies this tensor network
        by scalar inplace."""
        for core in self.cores:
            core.data *= scalar ** (1 / len(self.cores))
        return self


class ManyConvSBS(nn.Module):
    def __init__(
        self,
        in_num_channels: int,
        in_quantum_dim_size: int,
        bond_dim_size: int,
        trace_edge: bool,
        cores_specs: Tuple[SBSSpecCore, ...],
        initializations: Optional[
            Tuple[
                Union[
                    DumbNormalInitialization,
                    KhrulkovNormalInitialization,
                    NormalPreservingOutputStdInitialization,
                    MinRandomEyeInitialization,
                ],
                ...,
            ]
        ] = None,
    ):
        """If initializations is None, default initialization of ConvSBS is used."""
        super().__init__()
        if initializations is not None:
            assert len(initializations) == len(cores_specs)

        strings_specs = tuple(
            SBSSpecString(
                cores_spec,
                (bond_dim_size if trace_edge else 1,)
                + (bond_dim_size,) * (len(cores_spec) - 1),
                in_num_channels,
                in_quantum_dim_size,
            )
            for cores_spec in cores_specs
        )

        output_quantum_dim_sizes = tuple(
            string_spec.out_total_quantum_dim_size for string_spec in strings_specs
        )
        assert all(
            size == output_quantum_dim_sizes[0] for size in output_quantum_dim_sizes[1:]
        )

        if initializations is None:
            self.strings = nn.ModuleList([ConvSBS(spec) for spec in strings_specs])
        else:
            self.strings = nn.ModuleList(
                [
                    ConvSBS(spec, initialization)
                    for (spec, initialization) in zip(strings_specs, initializations)
                ]
            )

    def forward(
        self, channels: Union[torch.Tensor, Tuple[torch.Tensor, ...]], /
    ) -> Tuple[torch.Tensor, ...]:
        return tuple(module(channels) for module in self.strings)
