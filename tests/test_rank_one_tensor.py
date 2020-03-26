import pytest

import torch

from dctn.rank_one_tensor import RankOneTensorsBatch


def test_basic_methods() -> None:
  x = RankOneTensorsBatch(
    array=torch.Tensor(
      [[[[1.0], [2.0]], [[2.0], [-3.0]]], [[[4.0], [2.0]], [[-5.0], [-10.0]]]]
    ),
    factors_dim=1,
    coordinates_dim=2,
  )

  assert x.batch_shape == (2, 1)
  assert x.ntensors == 2
  assert x.ncoordinates == 4

  assert x.sum_per_tensor().shape == (2, 1)
  assert torch.allclose(x.sum_per_tensor(), torch.Tensor([[-3.0], [-90.0]]))

  assert x.sum_over_batch().ndim == 0
  assert torch.allclose(x.sum_over_batch(), torch.tensor(-93.0))

  assert x.mean_per_tensor().shape == (2, 1)
  assert torch.allclose(x.mean_per_tensor(), torch.Tensor([[-0.75], [-22.5]]))

  assert x.mean_over_batch().ndim == 0
  assert torch.allclose(x.mean_over_batch(), torch.tensor(-11.625))

  assert x.squared_fro_norm_per_tensor().shape == (2, 1)
  assert torch.allclose(
    x.squared_fro_norm_per_tensor(), torch.Tensor([[65.], [2500.]])
  )

  assert x.squared_fro_norm_over_batch().ndim == 0
  assert torch.allclose(x.squared_fro_norm_over_batch(), torch.tensor(2565.))

  assert x.var_over_batch().ndim == 0
  assert torch.allclose(x.var_over_batch(), torch.tensor(211.9821))

  assert x.std_over_batch().ndim == 0
  assert torch.allclose(x.std_over_batch(), torch.tensor(14.5596))
