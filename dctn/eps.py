import itertools
import logging
import functools
import math
from typing import Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import opt_einsum as oe

from dctn.align import align
from dctn.contraction_path_cache import contract


def eps(core: Tensor, input: Tensor) -> Tensor:
    num_channels, batch_size, height, width, in_size = input.shape
    kernel_size = math.isqrt((core.ndim - 1) // num_channels)
    assert core.shape[:-1] == tuple(in_size for _ in range(kernel_size ** 2 * num_channels))
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


def eps_one_by_one(core: Tensor, input: Tensor) -> Tensor:
    num_channels, batch_size, height, width, in_size = input.shape
    kernel_size = math.isqrt((core.ndim - 1) // num_channels)
    assert core.shape[:-1] == tuple(in_size for _ in range(kernel_size ** 2 * num_channels))
    out_size = core.shape[-1]
    aligned_input_cores = align(input, kernel_size)
    first = True
    for input_core in aligned_input_cores:
        if first:
            intermediate = torch.einsum("bhwi,i...->bhw...", input_core, core)
            first = False
        else:
            intermediate = torch.einsum("bhwi,bhwi...->bhw...", input_core, intermediate)

    assert intermediate.shape == (
        batch_size,
        height - kernel_size + 1,
        width - kernel_size + 1,
        out_size,
    )
    return intermediate


def calc_eps_shape(
    kernel_size: int, in_num_channels: int, in_size: int, out_size: int
) -> Tuple[int, ...]:
    """Calculates what shape an EPS tensor with these parameters must have."""
    return (in_size,) * (kernel_size ** 2 * in_num_channels) + (out_size,)


class EPS(nn.Module):
    def __init__(self, kernel_size: int, in_num_channels: int, in_size: int, out_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_num_channels = in_num_channels
        self.in_size = in_size
        self.out_size = out_size
        self.core = nn.Parameter(
            make_eps_unit_theoretical_output_std(
                kernel_size,
                in_num_channels,
                in_size,
                out_size,
                torch.device("cpu"),
                torch.float32,
            )
        )

    @property
    def matrix_shape(self) -> Tuple[int, int]:
        return matrix_shape(self.core)

    def forward(self, input: Tensor) -> Tensor:
        return eps(self.core, input)


def matrix_shape(eps_core: Tensor) -> Tuple[int, int]:
    assert is_eps(eps_core)
    out_size = eps_core.shape[-1]
    in_total_size = math.prod(eps_core.shape[:-1])
    return out_size, in_total_size


def contract_on_input_dims(a: Tensor, b: Tensor) -> Tensor:
    """result.shape: (out dim of a, out dim of b)."""
    assert is_eps(a)
    assert is_eps(b)
    a_out_dim_size = a.shape[-1]
    b_out_dim_size = b.shape[-1]
    return a.reshape(-1, a_out_dim_size).T @ b.reshape(-1, b_out_dim_size)


def is_eps(a: Tensor) -> bool:
    """Returns whether a can plausibly be an EPS, judging by its shape."""
    return a.ndim >= 2 and all(dim_size == a.shape[0] for dim_size in a.shape[:-1])


def inner_product(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert is_eps(a)
    return torch.dot(a.reshape(-1), b.reshape(-1))


@torch.no_grad()
def transform_in_slices(eps_core: Tensor, x: Tensor, batch_size: int) -> torch.Tensor:
    """Given `x` of shape (num_channels, dataset_size, height, width, in_size), transforms it
    with `eps_core` to get an output of shape (1, dataset_size, new_height, new_width, out_size).
    Doesn't propagate grad. Applies in batches to save memory, so can be used even with large
    dataset_size.

    This is useful for transforming a large part of a dataset."""
    assert is_eps(eps_core)
    return torch.cat(
        tuple(eps(eps_core, x_slice) for x_slice in x.split(batch_size, dim=1))
    ).unsqueeze(0)


def total_in_dim_size(kernel_size: int, in_num_channels: int, in_size: int) -> int:
    return in_size ** (in_num_channels * kernel_size ** 2)


def make_eps_unit_theoretical_output_std(
    kernel_size: int,
    in_num_channels: int,
    in_size: int,
    out_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    std = (
        total_in_dim_size(kernel_size, in_num_channels, in_size) ** -0.5
    )  # preserves std during forward pass
    return std * torch.randn(
        *calc_eps_shape(kernel_size, in_num_channels, in_size, out_size), dtype=dtype
    ).to(device)


def make_eps_unit_empirical_output_std(
    kernel_size: int,
    out_size: int,
    input: Tensor,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> Tensor:
    num_channels, dataset_size, height, width, in_size = input.shape
    core = torch.randn(
        *(in_size,) * (kernel_size ** 2 * num_channels), out_size, dtype=dtype
    ).to(device)
    output = transform_in_slices(core, input.to(device, dtype), batch_size)
    core /= output.std(unbiased=False)
    logging.getLogger(__name__).info(
        f"Initialized an EPS with empirical std = {core.std(unbiased=False)}"
    )
    return core
