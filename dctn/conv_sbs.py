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
from .digits_to_words import d2w, w2d

# here goes types of initialization of ConvSBS


@attrs(auto_attribs=True, frozen=True)
class DumbNormalInitialization:
    std_of_elements_of_cores: float


@attrs(auto_attribs=True, frozen=True)
class KhrulkovNormalInitialization:
    std_of_elements_of_matrix: Optional[float]


class ConvSBS(nn.Module):
    def __init__(
        self,
        spec: SBSSpecString,
        initialization: Union[
            DumbNormalInitialization, KhrulkovNormalInitialization
        ] = DumbNormalInitialization(0.9),
    ):
        super().__init__()
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
        self._first_stage_einsum_exprs = None
        self._second_stage_einsum_expr = None

    def init_khrulkov_normal(
        self, std_of_elements_of_matrix: Optional[float] = None
    ) -> None:
        logger = logging.getLogger(f"{__name__}.{self.init_khrulkov_normal.__qualname__}")
        if std_of_elements_of_matrix is not None:
            var_of_elements_of_matrix = std_of_elements_of_matrix ** 2
        else:
            # See Tensorized Embedding Layers for Efficient Model Compression by Khrulkov
            # Section "Initialization"
            matrix_num_columns = self.spec.in_quantum_dim_size ** (
                self.spec.in_num_channels * len(self.spec.cores)
            )
            matrix_num_rows = self.spec.out_total_quantum_dim_size
            var_of_elements_of_matrix = 2 / (matrix_num_columns + matrix_num_rows)
            logger.info(
                f"matrix_num_columns = {matrix_num_columns}, matrix_num_rows = {matrix_num_rows}, var_of_elements_of_matrix = {var_of_elements_of_matrix}"
            )

        prod_of_ranks = functools.reduce(operator.mul, self.spec.bond_sizes)
        var_of_cores_elements = var_of_elements_of_matrix ** (
            1 / len(self.cores)
        ) / prod_of_ranks ** (1 / len(self.cores))
        logger.info(
            f"bond_sizes = {self.spec.bond_sizes}, prod_of_ranks = {prod_of_ranks}, var_of_cores_elements = {var_of_cores_elements}"
        )
        for core in self.cores:
            torch.nn.init.normal_(core, std=math.sqrt(var_of_cores_elements))

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
                        f"out_quantum_{i}",
                        d2w(f"bond_{i}"),
                        d2w(f"bond_{i+1 if i < len(self.cores)-1 else 0}"),
                        "height",
                        "width",
                    ),
                )
                for i, shape in enumerate(self.spec.shapes)
            ),
            self._second_stage_result_dimensions_names,
            optimize="auto",
        )

    def forward(
        self, channels: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
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
                value=1.0
                # value=float("nan"),
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


class ManyConvSBS(nn.Module):
    def __init__(
        self,
        in_num_channels: int,
        in_quantum_dim_size: int,
        bond_dim_size: int,
        trace_edge: bool,
        cores_specs: Tuple[SBSSpecCore, ...],
        initializations: Optional[
            Tuple[Union[DumbNormalInitialization, KhrulkovNormalInitialization], ...]
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
        self, channels: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
    ) -> Tuple[torch.Tensor, ...]:
        return tuple(module(channels) for module in self.strings)
