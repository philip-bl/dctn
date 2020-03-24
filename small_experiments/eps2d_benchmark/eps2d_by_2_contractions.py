from dctn.benchmark import benchmark_torch

import torch

def create_tensors(device, dtype):
    param = torch.randn((2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2), device=device, dtype=dtype)
    inputs = [torch.randn((64, 25, 25, 2), device=device, dtype=dtype) for _ in range(16)]
    input_part_0 = torch.einsum("abti,abtj,abtk,abtl,abtm,abtn,abto,abtp->abtijklmnop", inputs[:8])
    assert input_part_0.is_contiguous()
    return input_part_0.requires_grad_(), param.requires_grad_()

def calc_intermediate_matmul(input_part_0, param):
    return input_part_0.reshape(64*25*25, 256) @ param.reshape(256, 2*2*2*2*2*2*2*2*2)

def calc_intermediate_einsum(input_part_0, param):
    return torch.einsum("bhwijklmnop,ijklmnop...->bhw...", input_part_0, param)

print(benchmark_torch(calc_intermediate_matmul, create_tensors, torch.float32, torch.device("cuda"), 50))
print(benchmark_torch(calc_intermediate_einsum, create_tensors, torch.float32, torch.device("cuda"), 50))


def create_tensors_for_complete_thing(device, dtype):
    param = torch.randn(
        (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2), device=device, dtype=dtype, requires_grad=True)
    inputs = [torch.randn((64, 25, 25, 2), device=device, dtype=dtype, requires_grad=True) for _ in range(16)]
    return (param, *inputs)

def do_the_thing(*args):
    param = args[0]
    inputs = args[1:]
    input_part_0 = torch.einsum("bhwi,bhwj,bhwk,bhwl,bhwm,bhwn,bhwo,bhwp->bhwijklmnop", inputs[:8])
    input_part_1 = torch.einsum("bhwi,bhwj,bhwk,bhwl,bhwm,bhwn,bhwo,bhwp->bhwijklmnop", inputs[8:])
    assert input_part_0.is_contiguous()
    assert input_part_1.is_contiguous()
    intermediate = calc_intermediate_einsum(input_part_0, param)
    return torch.einsum("bhwijklmnop,bhwijklmnopz->bhwz", input_part_1, intermediate)

print(benchmark_torch(do_the_thing, create_tensors_for_complete_thing, torch.float32, torch.device("cuda"), 500))
