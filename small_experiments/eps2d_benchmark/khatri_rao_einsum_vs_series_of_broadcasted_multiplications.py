import torch

from dctn.benchmark import benchmark_torch

import tensorly as tl
tl.set_backend("pytorch")

import einops


def create_tensors(device, dtype):
  return tuple(torch.randn((64, 25, 25, 2), device=device, dtype=dtype, requires_grad=True) for _ in range(8))


def khatri_rao_via_einsum(*tensors) -> torch.Tensor:
  letters = tuple(chr(i) for i in range(ord("d"), ord("d") + len(tensors)))
  subscripts = tuple(f"abc{letter}" for letter in letters)
  equation_lhs = ",".join(subscripts)
  equation_rhs = "abc" + "".join(letters)
  equation = equation_lhs + "->" + equation_rhs
  return torch.einsum(equation, *tensors)

def khatri_rao_via_tl(*tensors) -> torch.Tensor:
  matrices = tuple(tensor.reshape(-1, tensor.shape[-1]).T for tensor in tensors)
  result_unfolded = tl.tenalg.khatri_rao(matrices).T
  result = einops.rearrange(
    result_unfolded, "(a b c) (d e f g h i j k) -> a b c d e f g h i j k",
    a=64, b=25, c=25, d=2, e=2, f=2, g=2, h=2, i=2, j=2, k=2
  )
  return result


for func in (khatri_rao_via_tl, khatri_rao_via_einsum):
  print(benchmark_torch(
    func, create_tensors, torch.float64, torch.device("cuda"), 500
  ))
  print(f"{torch.cuda.memory_stats()['allocated_bytes.all.peak']=}")
  print()

