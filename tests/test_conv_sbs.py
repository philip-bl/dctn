import pytest

import torch

from dctn.conv_sbs_spec import SBSSpecCore, SBSSpecString, Pos2D
from dctn.conv_sbs import ConvSBS, KhrulkovNormalInitialization


@pytest.mark.parametrize(
  "num_monte_carlo_iterations, bond_size, in_num_channels, in_quantum_dim_size, trace_edge, desired_std, allowed_relative_error",
  (
    (1000, 5, 2, 2, False, 0.5, 0.3),
    (1500, 6, 2, 2, False, 100.0, 0.5),
    (1000, 12, 1, 4, False, 7.0, 0.5),
    (2000, 4, 2, 2, False, 1.0, 0.5),
    (1000, 8, 1, 2, True, 0.1, 0.5),
  ),
)
def test_khrulkov_normal_init_and_std(
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
        SBSSpecCore(Pos2D(0, 0), 1),
        SBSSpecCore(Pos2D(0, 1), 1),
        SBSSpecCore(Pos2D(0, 2), 1),
        SBSSpecCore(Pos2D(1, 2), 1),
        SBSSpecCore(Pos2D(1, 1), 2),
        SBSSpecCore(Pos2D(1, 0), 1),
        SBSSpecCore(Pos2D(2, 0), 1),
        SBSSpecCore(Pos2D(2, 1), 1),
        SBSSpecCore(Pos2D(2, 2), 1),
      ),
      (bond_size if trace_edge else 1,) + (bond_size,) * 8,
      in_num_channels,
      in_quantum_dim_size,
    )

    conv_sbs = ConvSBS(spec, KhrulkovNormalInitialization(desired_std))
    std_fast = conv_sbs.var() ** 0.5
    empiric_stds.append(std_fast.item())
    if i == 0:
      explicit_tensor = conv_sbs.as_explicit_tensor()
      std_slow = explicit_tensor.std()
      rtol = 1e-3
      assert torch.allclose(std_slow, std_fast, rtol=rtol)
      assert torch.allclose(explicit_tensor.mean(), conv_sbs.mean(), rtol=rtol)
      assert torch.allclose(
        explicit_tensor.norm("fro"), conv_sbs.fro_norm(), rtol=rtol
      )

  mean_empiric_std = sum(empiric_stds) / len(empiric_stds)
  relative_error = abs(desired_std - mean_empiric_std) / desired_std
  assert relative_error <= allowed_relative_error
