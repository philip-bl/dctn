import itertools
from typing import *
import torch
from einops import rearrange
import opt_einsum as oe
from dctn.eps import eps2d_simple, eps2d_oe, eps2d_oe_via_padding, align, align_via_padding


def test_eps2d_single_pixel_output() -> None:
    input = torch.randn((2, 3, 2, 2, 2), dtype=torch.float64)
    core = torch.rand((*(2 for _ in range(8)), 4), dtype=torch.float64)
    eps2d_result = rearrange(eps2d_simple(core, input), "b () () o -> b o")
    assert eps2d_result.shape == (3, 4)
    oe_result = oe.contract(
        "01234567θ,b0,b1,b2,b3,b4,b5,b6,b7->bθ",
        core,
        input[0,:,0,0], input[1,:,0,0],
        input[0,:,0,1], input[1,:,0,1],
        input[0,:,1,0], input[1,:,1,0],
        input[0,:,1,1], input[1,:,1,1]
    )
    assert torch.allclose(eps2d_result, oe_result)


def test_eps2d_two_pixels_output() -> None:
    input = torch.randn((1, 1, 4, 3, 2), dtype=torch.float64)
    core = torch.rand((*(2 for _ in range(9)), 4), dtype=torch.float64)
    eps2d_result = eps2d_simple(core, input)
    assert eps2d_result.shape == (1, 2, 1, 4)
    oe_result_0 = oe.contract(
        "012345678θ,0,1,2,3,4,5,6,7,8->θ",
        core,
        input[0,0,0,0],input[0,0,0,1],input[0,0,0,2],
        input[0,0,1,0],input[0,0,1,1],input[0,0,1,2],
        input[0,0,2,0],input[0,0,2,1],input[0,0,2,2]
    )
    assert torch.allclose(eps2d_result[0, 0, 0], oe_result_0)
    oe_result_1 = oe.contract(
        "012345678θ,0,1,2,3,4,5,6,7,8->θ",
        core,
        input[0,0,1,0],input[0,0,1,1],input[0,0,1,2],
        input[0,0,2,0],input[0,0,2,1],input[0,0,2,2],
        input[0,0,3,0],input[0,0,3,1],input[0,0,3,2]
    )
    assert torch.allclose(eps2d_result[0, 1, 0], oe_result_1)


def test_eps_slicing_and_padding_versions_give_the_same_output() -> None:
    # create data
    kernel_size = 2
    num_channels = 2
    core = torch.randn(*(2,) * kernel_size**2 * num_channels, 10, requires_grad=True, device="cpu", dtype=torch.float64)
    input = torch.randn(num_channels, 64, 28, 28, 2, requires_grad=True, device="cpu", dtype=torch.float64)

    # check that aligning the results works the same way
    aligned_via_slicing = tuple(align(input, kernel_size))
    aligned_via_padding, good_h_slice, good_w_slice = align_via_padding(input, kernel_size)
    aligned_via_padding = tuple(aligned_via_padding)
    grad_output = torch.randn_like(aligned_via_slicing[0])
    for i, (sl, pa) in enumerate(zip(aligned_via_slicing, aligned_via_padding)):
        pa_good = pa[:, good_h_slice, good_w_slice]
        assert torch.all(sl == pa_good)
        channel = i % num_channels
        δw = (i // num_channels) % kernel_size
        δh = i // (num_channels * kernel_size)
        assert torch.all(pa_good == input[channel, :, δh : 28-kernel_size+δh+1, δw : 28-kernel_size+δw+1])
        assert torch.all(sl == input[channel, :, δh : 28-kernel_size+δh+1, δw : 28-kernel_size+δw+1])
        sl.backward(grad_output)
        input_grad_slicing = input.grad.clone()
        input.grad.zero_()
        pa_good.backward(grad_output)
        input_grad_padding = input.grad.clone()
        input.grad.zero_()
        assert torch.all(input_grad_slicing == input_grad_padding) # is very different - wtf?


    result_via_slicing = eps2d_oe(core, input)
    grad_output = torch.randn_like(result_via_slicing)
    result_via_slicing.backward(grad_output)
    core_grad_slicing = core.grad.clone()
    core.grad.zero_()
    input_grad_slicing = input.grad.clone()
    input.grad.zero_()

    result_via_padding = eps2d_oe_via_padding(core, input)
    assert torch.all(result_via_slicing == result_via_padding)
    result_via_padding.backward(grad_output)
    core_grad_padding = core.grad.clone()
    core.grad.zero_()
    input_grad_padding = input.grad.clone()
    input.grad.zero_()

    assert torch.allclose(core_grad_padding, core_grad_slicing)
    assert torch.allclose(input_grad_padding, input_grad_slicing)
    
