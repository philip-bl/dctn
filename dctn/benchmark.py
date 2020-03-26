from timeit import timeit

from typing import *

import torch
from torch import Tensor


def synchronize_if_cuda(device) -> None:
  if device.type == "cuda":
    torch.cuda.synchronize(device)


def benchmark_torch(
  func,
  args_creator: Callable[[Any, Any], Iterable[Any]],
  dtype,
  device,
  num_iterations,
) -> Dict[str, Any]:
  """`args_creator` must set requires_grad where you want it."""
  args = args_creator(dtype=dtype, device=device)

  # warm up for benchmarking forward pass
  with torch.no_grad():
    func(*args)
  synchronize_if_cuda(device)

  # benchmark forward pass
  @torch.no_grad()
  def run_forward() -> None:
    for i in range(num_iterations):
      func(*args)
    synchronize_if_cuda(device)

  forward_total_sec = timeit(run_forward, number=1)

  # warm up for benchmarking forward+backward passes
  output = func(*args)
  out_grad = torch.randn_like(output)
  output.backward(out_grad)
  synchronize_if_cuda(device)

  def run_forward_backward() -> None:
    for i in range(num_iterations):
      func(*args).backward(out_grad)
    synchronize_if_cuda(device)

  forward_backward_total_sec = timeit(run_forward_backward, number=1)

  result = {
    "func": func.__name__,
    "forward_seconds_per_iteration": forward_total_sec / num_iterations,
    "forward_backward_seconds_per_iteration": forward_backward_total_sec
    / num_iterations,
    "device": "cpu" if device.type == "cpu" else torch.cuda.get_device_name(device),
    "dtype": str(dtype),
    "num_iterations": num_iterations,
    "args_creator": str(args_creator),
  }
  return result
