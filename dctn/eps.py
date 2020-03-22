import itertools
import functools
import math
from typing import *

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


def eps2d(core: Tensor, input: Tensor) -> Tensor:
    num_channels, batch_size, height, width, in_size = input.shape
    kernel_size = math.isqrt((core.ndim - 1) // num_channels)
    assert core.shape[:-1] == tuple(
        in_size for _ in range(kernel_size ** 2 * num_channels)
    )
    out_size = core.shape[-1]
    first = True
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
            input_core = input[channel, :, height_slice, width_slice]
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
