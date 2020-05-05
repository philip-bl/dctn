import itertools
from typing import Union, Tuple, Iterable

import torch
from torch import Tensor

from .pos2d import Pos2D
from .rank_one_tensor import RankOneTensorsBatch


def align_with_positions(
    input: Union[Tensor, Tuple[Tensor, ...]], positions: Tuple[Pos2D, ...]
) -> Iterable[Tensor]:
    num_channels = len(input)
    batch_size, height, width, in_size = input[0].shape
    max_h = max(pos.h for pos in positions)
    max_w = max(pos.w for pos in positions)
    assert min(pos.h for pos in positions) == 0
    assert min(pos.w for pos in positions) == 0
    for pos in positions:
        height_slice = slice(
            pos.h,
            None
            if (unused_height_on_bottom := max_h - pos.h) == 0
            else -unused_height_on_bottom,
        )
        width_slice = slice(
            pos.w,
            None if (unused_width_on_right := max_w - pos.w) == 0 else -unused_width_on_right,
        )
        for channel in range(num_channels):
            yield input[channel][:, height_slice, width_slice]


def align(input: Tensor, kernel_size: int) -> Iterable[Tensor]:
    """For kernel_size=3, the order goes like this:
  0 1 2
  3 4 5
  6 7 8"""
    return align_with_positions(
        input,
        tuple(
            Pos2D(δh, δw)
            for (δh, δw) in itertools.product(range(kernel_size), range(kernel_size))
        ),
    )


def make_windows(x: torch.Tensor, kernel_size: int) -> RankOneTensorsBatch:
    """`x`: tensor of shape (num_channels, batch_size, height, width, in_quantum_size)."""
    return RankOneTensorsBatch(
        torch.cat(
            tuple(
                torch.stack(tuple(align(x_slice, kernel_size)), dim=0)
                for x_slice in x.split(128, dim=1)
            ),
            dim=1,
        ),
        factors_dim=0,
        coordinates_dim=4,
    )
