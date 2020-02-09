import pytest

import torch

from dctn.conv_sbs_spec import SBSSpecCore, SBSSpecString, Pos2D
from dctn.conv_sbs import ConvSBS, KhrulkovNormalInitialization


@pytest.mark.parametrize(
    "num_monte_carlo_iterations, bond_size, in_num_channels, in_quantum_dim_size, trace_edge, desired_std, allowed_relative_error",
    (
        (40, 5, 2, 2, False, 0.5, 0.3),
        (40, 1, 3, 3, True, 100.0, 0.3),
        (40, 20, 1, 4, False, 7.0, 0.3),
    ),
)
def test_one_combination(
    num_monte_carlo_iterations,
    bond_size,
    in_num_channels,
    in_quantum_dim_size,
    trace_edge,
    desired_std,
    allowed_relative_error,
) -> None:
    empiric_stds = []
    for i in range(num_monte_carlo_iterations):
        spec = SBSSpecString(
            (
                SBSSpecCore(Pos2D(-1, -1), 1),
                SBSSpecCore(Pos2D(-1, 0), 1),
                SBSSpecCore(Pos2D(-1, 1), 1),
                SBSSpecCore(Pos2D(0, 1), 1),
                SBSSpecCore(Pos2D(0, 0), 2),
                SBSSpecCore(Pos2D(0, -1), 1),
                SBSSpecCore(Pos2D(1, -1), 1),
                SBSSpecCore(Pos2D(1, 0), 1),
                SBSSpecCore(Pos2D(1, 1), 1),
            ),
            (bond_size if trace_edge else 1,) + (bond_size,) * 8,
            in_num_channels,
            in_quantum_dim_size,
        )

        conv_sbs = ConvSBS(spec, KhrulkovNormalInitialization(desired_std))
        std_fast = conv_sbs.var() ** 0.5
        std_slow = conv_sbs.as_explicit_tensor().std()
        assert torch.allclose(std_fast, std_slow)
        empiric_stds.append(std_fast.item())

    mean_empiric_std = sum(empiric_stds) / len(empiric_stds)
    relative_error = abs(desired_std - mean_empiric_std) / desired_std
    assert relative_error <= allowed_relative_error
