import functools
import operator
from itertools import chain, islice
from attr import attrs, attrib, evolve
from typing import *


@attrs(auto_attribs=True, frozen=True)
class Pos2D:
    h: int
    w: int


@attrs(auto_attribs=True, frozen=True)
class SBSSpecCore:
    position: Pos2D
    out_quantum_dim_size: int


@attrs(auto_attribs=True, frozen=True)
class SBSCoreShape:
    out_quantum_dim_size: int
    bond_left_size: int
    bond_right_size: int
    in_num_channels: int
    in_quantum_dim_size: int

    def as_tuple(self) -> Tuple[int, ...]:
        return (
            self.out_quantum_dim_size,
            self.bond_left_size,
            self.bond_right_size,
        ) + (self.in_quantum_dim_size,) * self.in_num_channels

    @property
    def dimensions_names(self) -> Tuple[str, ...]:
        return ("out_quantum", "bond_left", "bond_right") + tuple(
            f"in_quantum_{i}" for i in range(self.in_num_channels)
        )

    @property
    def total_dangling_dimensions_size(self) -> int:
        return in_quantum_dim_size ** in_num_channels * out_quantum_dim_size


@attrs(frozen=True)
class SBSSpecString:
    cores: Tuple[SBSSpecCore, ...] = attrib()
    bond_sizes: Tuple[int, ...] = attrib()
    in_num_channels: int = attrib()
    in_quantum_dim_size: int = attrib(default=2)

    @cores.validator
    def _check_positions(self, attribute, cores_value) -> None:
        if (
            not self.min_height_pos <= 0 <= self.max_height_pos
            or not self.min_width_pos <= 0 <= self.max_width_pos
        ):
            raise ValueError(f"Positions of cores are invalid")

    @bond_sizes.validator
    def _check_matching_lengths(self, attribute, bond_sizes_value) -> None:
        if len(bond_sizes_value) != len(self.cores):
            raise ValueError(
                f"The length of bond_sizes is {len(bond_sizes_value)}, it must be equal to the length of cores, which is {len(self.cores)}"
            )

    def __len__(self) -> int:
        return len(self.cores)

    @property
    def shapes(self) -> Tuple[SBSCoreShape, ...]:
        return tuple(
            SBSCoreShape(
                core.out_quantum_dim_size,
                bond_left_size,
                bond_right_size,
                self.in_num_channels,
                self.in_quantum_dim_size,
            )
            for (core, bond_left_size, bond_right_size) in zip(
                self.cores,
                self.bond_sizes,
                chain(islice(self.bond_sizes, 1, None), (self.bond_sizes[0],)),
            )
        )

    @property
    def min_height_pos(self) -> int:
        return min(core.position.h for core in self.cores)

    @property
    def max_height_pos(self) -> int:
        return max(core.position.h for core in self.cores)

    @property
    def min_width_pos(self) -> int:
        return min(core.position.w for core in self.cores)

    @property
    def max_width_pos(self) -> int:
        return max(core.position.w for core in self.cores)

    @property
    def out_total_quantum_dim_size(self) -> int:
        return functools.reduce(
            operator.mul, (core.out_quantum_dim_size for core in self.cores), 1
        )

    @property
    def nelement(self) -> int:
        """Returns the total number of elements in the TT tensor."""
        return functools.reduce(
            operator.mul, (core.total_dangling_dimensions_size for core in self.cores)
        )

    def get_dim_names(self, core_index: int) -> Tuple[str, ...]:
        """Returns dims names of core number core_index. These can be used in an einsum
expression."""

        return (
            f"out_quantum_{i}",
            "bond_{i}",
            f"bond_{i+1 if i < len(self)-1 else 0}",
            *(f"channel_{c}_{i}" for c in range(self.in_num_channels)),
        )

    @property
    def all_dim_names(self) -> Tuple[Tuple[str, ...], ...]:
        """Returns dimensions names of all cores. These can be used in an einsum expression.
The only names shared by cores will be bonds."""
        return tuple(self.get_dim_names(i) for i in range(len(self)))
