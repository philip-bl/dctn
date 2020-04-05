from itertools import permutations

import torch

from einops import rearrange

from dctn.pos2d import Pos2D
from dctn.conv_sbs_spec import SBSSpecCore, SBSSpecString
from dctn.conv_sbs import ConvSBS
from dctn.eps import eps


def test_conversion() -> None:
  in_num_channels = 2
  in_quantum_dim_size = 2
  batch_size = 3
  height = 4
  width = 5
  cores = (SBSSpecCore(Pos2D(0, 0), 1), SBSSpecCore(Pos2D(0, 1), 3),
                 SBSSpecCore(Pos2D(1, 0), 2), SBSSpecCore(Pos2D(1, 1), 4))
  for cores_permutation in permutations(cores):
    spec = SBSSpecString(cores_permutation, (3, 4, 5, 6), in_num_channels, in_quantum_dim_size)
    convsbs = ConvSBS(spec).double()
    with torch.no_grad():
      eps_tensor = convsbs.as_eps()
    assert eps_tensor.shape == (2,2, 2,2, 2,2, 2,2, 1*2*3*4)
    assert torch.all(eps_tensor == convsbs.as_eps())

    input = torch.randn(
      in_num_channels, batch_size, height, width, in_quantum_dim_size,
      dtype=torch.float64, requires_grad=True)

    convsbs_output = convsbs(input)
    out_grad = torch.randn_like(convsbs_output)
    convsbs_output.backward(out_grad)
    convsbs_input_grad = input.grad.clone()
    input.grad.zero_()

    eps_output = eps(eps_tensor, input)
    assert torch.allclose(eps_output, convsbs_output)
    eps_output.backward(out_grad)
    eps_input_grad = input.grad.clone()
    input.grad.zero_()
    assert torch.allclose(eps_input_grad, convsbs_input_grad)

