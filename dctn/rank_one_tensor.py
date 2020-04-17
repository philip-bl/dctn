"""This module provides functions for working with rank-one (not to confuse with order-one tensors aka vectors)
tensors. A rank-one tensor is a tensor representable as tensor product of vectors."""

import functools
import operator

from typing import Tuple

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

    @property
    def batch_shape(self) -> Tuple[int, ...]:
        return tuple(
            self.array.shape[i]
            for i in range(self.array.ndim)
            if i != self.factors_dim and i != self.coordinates_dim
        )

    @property
    def ncoordinates(self) -> int:
        """Returns the number of elements (aka coordinates) in one tensor.
        NOT IN THE WHOLE BATCH."""
        return self.array.shape[self.coordinates_dim] ** self.array.shape[self.factors_dim]

    @property
    def ntensors(self) -> int:
        """Returns the number of tensors in the batch."""
        return functools.reduce(operator.mul, self.batch_shape)

    def sum_per_tensor(self) -> torch.Tensor:
        """Returns, for each tensor in the batch, the sum of the elements of the tensor."""
        result = torch.prod(
            torch.sum(self.array, dim=self.coordinates_dim, keepdim=True),
            dim=self.factors_dim,
            keepdim=True,
        )
        squeezed_once = torch.squeeze(result, max(self.factors_dim, self.coordinates_dim))
        return torch.squeeze(squeezed_once, min(self.factors_dim, self.coordinates_dim))

    def sum_over_batch(self) -> torch.Tensor:
        """Returns the sum of the elements of all tensors in the batch."""
        return torch.sum(self.sum_per_tensor())

    def mean_per_tensor(self) -> torch.Tensor:
        """Returns, for each tensor in the batch, the mean of the elements of the tensor."""
        return self.sum_per_tensor() / self.ncoordinates

    def mean_over_batch(self) -> torch.Tensor:
        """Returns the mean of the elements of all tensors in the batch."""
        return self.sum_over_batch() / (self.ntensors * self.ncoordinates)

    def squared_fro_norm_per_tensor(self) -> torch.Tensor:
        """For each tensor in the batch, returns its squared Frobenius norm."""
        result = torch.prod(
            torch.norm(self.array, dim=self.coordinates_dim, keepdim=True) ** 2,
            dim=self.factors_dim,
            keepdim=True,
        )
        squeezed_once = torch.squeeze(result, max(self.factors_dim, self.coordinates_dim))
        return torch.squeeze(squeezed_once, min(self.factors_dim, self.coordinates_dim))

    def squared_fro_norm_over_batch(self) -> torch.Tensor:
        """Returns the squared Frobenius norm of the whole batch of tensors, which is
        the same as the sum of squared Frobenius norms of all tensors in the batch."""
        return torch.sum(self.squared_fro_norm_per_tensor())

    def var_over_batch(self, unbiased: bool = True) -> torch.Tensor:
        """Returns the empirical variance of the whole batch of tensors.
        Applies Bessel's correction iff unbiased is True."""
        sum = self.sum_over_batch()
        mean = self.mean_over_batch()
        nelement = self.ntensors * self.ncoordinates
        divisor = nelement - 1 if unbiased else nelement
        return (
            self.squared_fro_norm_over_batch() / divisor
            - 2 * sum / divisor * mean
            + nelement / divisor * mean ** 2
        )

    def std_over_batch(self, unbiased: bool = True) -> torch.Tensor:
        """Returns the empirical std of the whole batch of tensors.
        Applies Bessel's correction iff unbiased is True."""
        return self.var_over_batch() ** 0.5
