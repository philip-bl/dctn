import itertools
from typing import Tuple, Dict

import torch
from torch import Tensor

from einops import rearrange

from . import eps
from .contraction_path_cache import contract


def inner_product(epses1: Tuple[Tensor, ...], epses2: Tuple[Tensor, ...]) -> Tensor:
    """Calculates the inner product of epses1 and epses2. Each pair of corresponding EPSes must have the same
    kernel size, output size, and input size.
    Performs exactly what is describe in my notebook pages 6-6a."""
    assert len(epses1) == len(epses2)
    for eps1, eps2 in zip(epses1, epses2):
        assert eps1.shape == eps2.shape
        assert eps.is_eps(eps1)

    if len(epses1) == 1:  # 1) in my notebook page 6
        return eps.inner_product(epses1[0], epses2[0])

    # otherwise we do 2) in my notebook page 6
    # epses1 has epses: a, b, ...
    a, b = epses1[:2]
    # epses2 has epses: k, l, ...
    k = epses2[0]

    # step 1
    x = eps.contract_on_input_dims(a, k)
    # x.shape: (out dim of a, out dim of k).

    # step 2
    b_num_input_dims = b.ndim - 1
    # b is like (*in_dims, out_dim)
    # x is like (new_in_dim, old_in_dim)
    new_d = contract(
        b,
        tuple(f"in{i}" for i in range(b_num_input_dims)) + ("out",),
        *itertools.chain.from_iterable(
            (x, (f"in{i}", f"newin{i}")) for i in range(b_num_input_dims)
        ),
        tuple(f"newin{i}" for i in range(b_num_input_dims)) + ("out",),
    )
    assert eps.is_eps(new_d)
    return inner_product((new_d,) + epses1[2:], epses2[1:])


def specs_to_full_specs(
    epses_specs: Tuple[Tuple[int, int]], initial_in_size: int
) -> Tuple[Dict[str, int]]:
    """Each spec is a tuple representing kernel_size, out_size."""
    kernel_sizes = tuple(kernel_size for kernel_size, _ in epses_specs)
    out_sizes = tuple(out_size for _, out_size in epses_specs)
    in_sizes = (initial_in_size,) + out_sizes[:-1]
    return tuple(
        {
            "kernel_size": kernel_size,
            "in_num_channels": 1,
            "in_size": in_size,
            "out_size": out_size,
        }
        for kernel_size, out_size, in_size in zip(kernel_sizes, out_sizes, in_sizes)
    )


def make_epses_composition_unit_theoretical_output_std(
    epses_specs: Tuple[Tuple[int, int]],
    initial_in_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Tensor, ...]:
    return tuple(
        eps.make_eps_unit_theoretical_output_std(**full_spec, device=device, dtype=dtype)
        for full_spec in specs_to_full_specs(epses_specs, initial_in_size)
    )


def make_epses_composition_unit_empirical_output_std(
    epses_specs: Tuple[Tuple[int, int]],
    input: Tensor,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 128,
) -> Tuple[Tensor, ...]:
    epses = []
    for kernel_size, out_size in epses_specs:
        eps_core: Tensor = eps.make_eps_unit_empirical_output_std(
            kernel_size, out_size, input, device, dtype, batch_size
        )
        input = eps.transform_in_slices(eps_core, input.to(device, dtype), batch_size)
        epses.append(eps_core)
    return tuple(epses)


def contract_with_input(epses: Tuple[Tensor], input: Tensor) -> Tensor:
    """`input`: must have shape (channels, batch_size, height, width, quantum_in_dim).

    The returned value will have shape (batch_size, new_height, new_width, quantum_out_dim)."""
    assert all(eps.is_eps(tensor) for tensor in epses)
    intermediate: Tensor = input
    for eps_core in epses[:-1]:
        intermediate = rearrange(eps.eps(eps_core, intermediate), "b h w q -> () b h w q")
    return eps.eps(epses[-1], intermediate)


def epswise_squared_fro_norm(epses: Tuple[Tensor]) -> Tensor:
    assert all(eps.is_eps(tensor) for tensor in epses)
    return sum(eps_core.norm(p="fro") ** 2 for eps_core in epses)
