from functools import reduce
from typing import Callable, Tuple, Union, Sequence

from attr import attrs, attrib

import torch
from torch import Tensor


@torch.no_grad()
def transform_dataset(f: Callable[[Tensor], Tensor], x: Tensor, batch_size: int = 64):
    """`x` must be of shape channel×sample×height×width×quantum.
  `f` must be an `eps`-like function which takes one argument - input - and produces the output.
  Doesn't propagate gradient."""
    return torch.cat(tuple(f(slice) for slice in torch.split(x, batch_size, dim=1))).unsqueeze(
        0
    )


def implies(x: bool, y: bool) -> bool:
    return not x or y


def xor(*args: Tuple[bool]) -> bool:
    return reduce(lambda x, y: (x and not y) or (not x and y), args, False)


def exactly_one_true(*args: Tuple[bool]) -> bool:
    assert all(isinstance(arg, bool) for arg in args)
    return sum(args) == 1


@attrs(auto_attribs=True, frozen=True)
class ZeroCenteredNormalInitialization:
    std: float


@attrs(auto_attribs=True, frozen=True)
class ZeroCenteredUniformInitialization:
    maximum: float


@attrs(auto_attribs=True, frozen=True)
class FromFileInitialization:
    path: str


OneTensorInitialization = Union[
    ZeroCenteredNormalInitialization, ZeroCenteredUniformInitialization, FromFileInitialization
]


def raise_exception(exception):
    raise exception


def id_assert_shape_matches(tensor: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    assert tensor.shape == tuple(shape)
    return tensor
