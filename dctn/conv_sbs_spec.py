from itertools import chain, islice
from attr import attrs, attrib, evolve
from typing import *


@attrs(auto_attribs=True, frozen=True)
class SBSSpecCore:
    position: Tuple[int, int]
    out_quantum_dim_size: int


@attrs(auto_attribs=True, frozen=True)
class SBSCoreShape:
    out_quantum_dim_size: int
    bond_1_size: int
    bond_2_size: int
    in_num_channels: int
    in_quantum_dim_size: int

    def as_tuple(self) -> Tuple[int, ...]:
        return (self.out_quantum_dim_size, self.bond_1_size, self.bond_2_size) + (
            self.in_quantum_dim_size,
        ) * self.in_num_channels

    @property
    def dimensions_names(self) -> Tuple[str, ...]:
        return ("out_quantum", "bond_1", "bond_2") + tuple(
            f"in_quantum_{i}" for i in range(self.in_num_channels)
        )


@attrs(frozen=True)
class SBSSpecString:
    cores: Tuple[SBSSpecCore, ...] = attrib()
    bond_sizes: Tuple[int, ...] = attrib()
    in_num_channels: int = attrib()
    in_quantum_dim_size: int = attrib(default=2)

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
                bond_1_size,
                bond_2_size,
                self.in_num_channels,
                self.in_quantum_dim_size,
            )
            for (core, bond_1_size, bond_2_size) in zip(
                self.cores,
                self.bond_sizes,
                chain(islice(self.bond_sizes, 1, None), (self.bond_sizes[0],)),
            )
        )
