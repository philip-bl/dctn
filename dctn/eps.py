import itertools
import functools
import math
from typing import *

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import opt_einsum as oe


def align(input: Tensor, kernel_size: int) -> Iterable[Tensor]:
    """For kernel_size=3, the order goes like this:
    0 1 2
    3 4 5
    6 7 8"""
    num_channels, batch_size, height, width, in_size = input.shape
    for (δh, δw) in itertools.product(range(kernel_size), range(kernel_size)):
        # product goes like (0, 0), (0, 1), (0, 2), (1, 0), ...
        height_slice = slice(
            δh,
            None
            if (unused_height_on_bottom := kernel_size - δh - 1) == 0
            else -unused_height_on_bottom,
        )
        width_slice = slice(
            δw,
            None
            if (unused_width_on_right := kernel_size - δw - 1) == 0
            else -unused_width_on_right,
        )
        for channel in range(num_channels):
            yield input[channel, :, height_slice, width_slice]


def align_via_padding(input: Tensor, kernel_size: int) -> Tuple[Iterable[Tensor], slice, slice]:
    """For kernel_size=3, the order goes like this:
    0 1 2
    3 4 5
    6 7 8

    Apart from the actual tensors, returns two slices: representing the good range of height and
    the good range of width."""
    num_channels, batch_size, height, width, in_size = input.shape
    result = (
        F.pad(input[channel],
            (0, 0, pad_left := kernel_size-δw-1, kernel_size-1-pad_left,
                   pad_up   := kernel_size-δh-1, kernel_size-1-pad_up),
            mode="constant")
        for (δh, δw, channel)
        in itertools.product(range(kernel_size), range(kernel_size), range(num_channels))
    )
    return (result, slice(kernel_size - 1, height), slice(kernel_size - 1, width))


def eps2d_oe(core: Tensor, input: Tensor) -> Tensor:
    num_channels, batch_size, height, width, in_size = input.shape
    kernel_size = math.isqrt((core.ndim - 1) // num_channels)
    assert core.shape[:-1] == tuple(
        in_size for _ in range(kernel_size ** 2 * num_channels)
    )
    out_size = core.shape[-1]
    aligned_input_cores = tuple(align(input, kernel_size))
    contraction_path = (
        tuple(range(input_part_0_len := math.ceil(len(aligned_input_cores) / 2))),
        tuple(range(len(aligned_input_cores) - input_part_0_len)),
        (0, 1),
        (0, 1),
    )
    return oe.contract(
        *itertools.chain.from_iterable(
            (input_core, ("batch", "height", "width", f"in{index}"))
            for index, input_core in enumerate(aligned_input_cores)
        ),
        core,
        tuple(f"in{index}" for index in range(len(aligned_input_cores))) + ("out",),
        ("batch", "height", "width", "out"),
        optimize=contraction_path,
    )


def eps2d_simple(core: Tensor, input: Tensor) -> Tensor:
    num_channels, batch_size, height, width, in_size = input.shape
    kernel_size = math.isqrt((core.ndim - 1) // num_channels)
    assert core.shape[:-1] == tuple(
        in_size for _ in range(kernel_size ** 2 * num_channels)
    )
    out_size = core.shape[-1]
    aligned_input_cores = align(input, kernel_size)
    first = True
    for input_core in aligned_input_cores:
        if first:
            intermediate = torch.einsum("bhwi,i...->bhw...", input_core, core)
            first = False
        else:
            intermediate = torch.einsum(
                "bhwi,bhwi...->bhw...", input_core, intermediate
            )

    assert intermediate.shape == (
        batch_size,
        height - kernel_size + 1,
        width - kernel_size + 1,
        out_size,
    )
    return intermediate
