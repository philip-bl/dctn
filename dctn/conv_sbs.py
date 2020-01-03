from itertools import chain

from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F

import opt_einsum as oe

import einops

from .conv_sbs_spec import SBSSpecString
from .digits_to_words import d2w, w2d


class ConvSBS(nn.Module):
    def __init__(self, spec: SBSSpecString):
        super().__init__()
        self.spec = spec
        self.cores = nn.ParameterList(
            (
                nn.Parameter(torch.randn(*shape.as_tuple()))
                for shape in self.spec.shapes
            )
        )
        self._first_stage_einsum_exprs = None
        self._second_stage_einsum_expr = None

    @property
    def _second_stage_result_dimensions_names(self) -> Tuple[str, ...]:
        return (
            "batch",
            *(f"out_quantum_{i}" for i in range(len(self.cores))),
            "height",
            "width",
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
    ) -> Tuple[torch.Tensor, ...]:
        """If passing a tensor, the very first dimension MUST be channels."""
        if isinstance(channels, torch.Tensor):
            channels = tuple(channels)
        # now channels is a tuple of tensors, each tensor corresponding to a channel
        # TODO I've started removing the usage of named tensors everywhere,
        # stopped here
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
        # padded_result = (
        #     self._second_stage_einsum_expr(*padded)
        #     .rename(*(d2w(name) for name in self._second_stage_result_dimensions_names))
        #     .flatten(
        #         [
        #             d2w(name)
        #             for name in self._second_stage_result_dimensions_names
        #             if "quantum" in name
        #         ],
        #         "quantum",
        #     )
        # )
        padded_result = einops.rearrange(
            self._second_stage_einsum_expr(*padded),
            "b {0} h w -> b ({0}) h w".format(
                " ".join((f"q{i}" for i in range(len(self.cores))))
            ),
        )

        # the good region is the region without NaNs
        good_region_height_limits = (
            self.spec.max_height_pos - self.spec.min_height_pos,
            padded_result.shape[-2]
            - (self.spec.max_height_pos - self.spec.min_height_pos),
        )
        good_region_width_limits = (
            self.spec.max_width_pos - self.spec.min_width_pos,
            padded_result.shape[-1]
            - (self.spec.max_width_pos - self.spec.min_width_pos),
        )
        # assert torch.all(
        #     torch.isnan(padded_result[:, :, : good_region_height_limits[0]])
        # )
        # assert torch.all(
        #     torch.isnan(padded_result[:, :, good_region_height_limits[1] :])
        # )
        # assert torch.all(
        #     torch.isnan(padded_result[:, :, :, : good_region_width_limits[0]])
        # )
        # assert torch.all(
        #     torch.isnan(padded_result[:, :, :, good_region_width_limits[1] :])
        # )
        result = padded_result[
            :,
            :,
            good_region_height_limits[0] : good_region_height_limits[1],
            good_region_width_limits[0] : good_region_width_limits[1],
        ]
        # assert torch.all(torch.isfinite(result))
        return result
