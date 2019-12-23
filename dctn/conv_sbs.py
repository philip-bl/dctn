from itertools import chain

from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F

import opt_einsum as oe

from .conv_sbs_spec import SBSSpecString


class ConvSBS(nn.Module):
    def __init__(self, spec: SBSSpecString):
        super().__init__()
        self.spec = spec
        self.cores = nn.ParameterList()  # TODO actually create the cores
        self._first_stage_einsum_exprs = None

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
                    "height",
                    "width",
                    "out_quantum",
                    "bond_1",
                    "bond_2",
                ),  # the result
                optimize="optimal",
            )
            for shape in self.spec.shapes
        )

    def forward(
        self, channels: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
    ) -> Tuple[torch.Tensor, ...]:
        if isinstance(channels, torch.Tensor):
            channels = tuple(channels.align_to(["channel", ...]))
        raise NotImplementedError()  # TODO implement this
