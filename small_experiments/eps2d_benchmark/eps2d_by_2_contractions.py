import torch
import opt_einsum as oe
from dctn.benchmark import benchmark_torch


def create_tensors(device, dtype):
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
    intermediate = torch.einsum("bhwijklmnop,ijklmnop...->bhw...", input_part_0, param)
    return torch.einsum("bhwijklmnop,bhwijklmnopz->bhwz", input_part_1, intermediate)


def do_the_thing_oe(*args):
    param = args[0]
    inputs = args[1:]
    return oe.contract(
        "abci,abcj,abck,abcl,abcm,abcn,abco,abcp,"
        "abcq,abcr,abcs,abct,abcu,abcv,abcw,abcx,"
        "ijklmnopqrstuvwxy->abcy", *inputs, param,
        optimize=((0, 1, 2, 3, 4, 5, 6, 7), (0, 1, 2, 3, 4, 5, 6, 7), (0, 1), (0, 1)))    
    

print(benchmark_torch(do_the_thing, create_tensors, torch.float32, torch.device("cuda"), 500))
print(benchmark_torch(do_the_thing_oe, create_tensors, torch.float32, torch.device("cuda"), 500))
# do_the_thing_oe's forward is 15% slower, but it's forward+backward is 15%-20% faster
