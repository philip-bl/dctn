"""This module provides functions for working with rank-one (not to confuse with order-one tensors aka vectors)
tensors. A rank-one tensor is a tensor representable as tensor product of vectors."""

from typing import *

from attr import attrs, attrib

import torch


@attrs(frozen=True)
class RankOneTensorsBatch:
    """This class works with batches of rank-one tensors, such that each tensor is already
    represented as its tensor product factors, and all factors are of equal size. That is, the whole batch
    of rank-one tensors is stored as one multidimensional array.

    For each combination of indices corresponding to dimensions other than factors_dim and coordinates_dim,
    the slice of array corresponding to these indices is a 2d array, and it contains the factors of a rank-one
    tensor stored as its fibers.
    Fix index of the factors_dim dimension to select one factor of this tensor.
    Fix index of the coordinates_dim dimension to select one coordinate of the factor(s)."""

    array: torch.Tensor = attrib()
    factors_dim: int = attrib()
    coordinates_dim: int = attrib()

    @coordinates_dim.validator
    def _check_dims(self, _, __) -> None:
        assert self.factors_dim != self.coordinates_dim
        assert (
            0 <= self.factors_dim < self.array.ndim
            and 0 <= self.coordinates_dim < self.array.ndim
        )

    def sum(self) -> torch.Tensor:
        result = torch.prod(
            torch.sum(self.array, dim=self.coordinates_dim, keepdim=True),
            dim=self.factors_dim,
            keepdim=True,
        )
        squeezed_once = torch.squeeze(
            result, max(self.factors_dim, self.coordinates_dim)
        )
        return torch.squeeze(squeezed_once, min(self.factors_dim, self.coordinates_dim))
