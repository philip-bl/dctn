import itertools
from typing import *
import torch
from einops import rearrange
import opt_einsum as oe
from dctn.eps import eps_one_by_one, align


def test_eps_single_pixel_output() -> None:
  input = torch.randn((2, 3, 2, 2, 2), dtype=torch.float64)
  core = torch.rand((*(2 for _ in range(8)), 4), dtype=torch.float64)
  eps_result = rearrange(eps_one_by_one(core, input), "b () () o -> b o")
  assert eps_result.shape == (3, 4)
  oe_result = oe.contract(
    "01234567θ,b0,b1,b2,b3,b4,b5,b6,b7->bθ",
    core,
    input[0,:,0,0], input[1,:,0,0],
    input[0,:,0,1], input[1,:,0,1],
    input[0,:,1,0], input[1,:,1,0],
    input[0,:,1,1], input[1,:,1,1]
  )
  assert torch.allclose(eps_result, oe_result)


def test_eps_two_pixels_output() -> None:
  input = torch.randn((1, 1, 4, 3, 2), dtype=torch.float64)
  core = torch.rand((*(2 for _ in range(9)), 4), dtype=torch.float64)
  eps_result = eps_one_by_one(core, input)
  assert eps_result.shape == (1, 2, 1, 4)
  oe_result_0 = oe.contract(
    "012345678θ,0,1,2,3,4,5,6,7,8->θ",
    core,
    input[0,0,0,0],input[0,0,0,1],input[0,0,0,2],
    input[0,0,1,0],input[0,0,1,1],input[0,0,1,2],
    input[0,0,2,0],input[0,0,2,1],input[0,0,2,2]
  )
  assert torch.allclose(eps_result[0, 0, 0], oe_result_0)
  oe_result_1 = oe.contract(
    "012345678θ,0,1,2,3,4,5,6,7,8->θ",
    core,
    input[0,0,1,0],input[0,0,1,1],input[0,0,1,2],
    input[0,0,2,0],input[0,0,2,1],input[0,0,2,2],
    input[0,0,3,0],input[0,0,3,1],input[0,0,3,2]
  )
  assert torch.allclose(eps_result[0, 1, 0], oe_result_1)
