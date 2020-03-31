import torch

from einops import rearrange

from dctn.conv_sbs_spec import Pos2D, SBSSpecCore, SBSSpecString
from dctn.conv_sbs import ConvSBS
from dctn.eps import eps


def test_simple_conversion() -> None:
  """Test conversion when the ConvSBS has kernel_size*kernel_size cores in the most
  straightforward order."""
  in_num_channels = 2
  in_quantum_dim_size = 2
  batch_size = 6
  height = 8
  width = 9
  spec = SBSSpecString(
    (SBSSpecCore(Pos2D(0, 0), 1), SBSSpecCore(Pos2D(0, 1), 1),
     SBSSpecCore(Pos2D(1, 0), 4), SBSSpecCore(Pos2D(1, 1), 1)),
    (5, 5, 5, 5), in_num_channels, in_quantum_dim_size)
  convsbs = ConvSBS(spec).double()
  eps_tensor = rearrange(
    convsbs.as_explicit_tensor().detach(), "... () () outquantum () -> ... outquantum")
  assert eps_tensor.shape == (2,2, 2,2, 2,2, 2,2, 4)

  # input = torch.randn(
  #   in_num_channels, batch_size, height, width, in_quantum_dim_size,
  #   dtype=torch.float64, requires_grad=True)
  input = torch.ones(
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
