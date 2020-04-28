import itertools
from typing import Tuple

import torch
from torch import Tensor

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
