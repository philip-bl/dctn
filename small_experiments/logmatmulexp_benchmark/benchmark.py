# TODO refactor this file to use dctn/benchmark.py
import os
import itertools
from functools import reduce
from timeit import timeit

from typing import Dict, Tuple, Any

import torch

from libcrap import save_json, load_json

from dctn.logmatmulexp import logmatmulexp, logmatmulexp_lowmem


def synchronize_if_cuda(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(dim_size: int, num_matrices: int, dtype, device, func, num_iterations):
    # preparation
    matrices = tuple(
        torch.randn((dim_size, dim_size), dtype=dtype, device=device)
        for i in range(num_matrices)
    )

    # benchmark forward pass
    with torch.no_grad():
        reduce(func, matrices)  # warm up
    synchronize_if_cuda(device)

    @torch.no_grad()
    def run_forward() -> None:
        for i in range(num_iterations):
            reduce(func, matrices)
        synchronize_if_cuda(device)

    forward_total_sec = timeit(run_forward, number=1)

    # benchmark forward+backward pass
    matrices[0].requires_grad_()
    out_grad = torch.ones(dim_size, dim_size, dtype=dtype, device=device)
    reduce(func, matrices).backward(out_grad)
    synchronize_if_cuda(device)

    def run_forward_backward() -> None:
        for i in range(num_iterations):
            reduce(func, matrices).backward(out_grad)
        synchronize_if_cuda(device)

    forward_backward_total_sec = timeit(run_forward_backward, number=1)

    result = {
        "func": func.__name__,
        "forward_seconds_per_iteration": forward_total_sec / num_iterations,
        "forward_backward_seconds_per_iteration": forward_backward_total_sec / num_iterations,
        "device": "cpu" if device.type == "cpu" else torch.cuda.get_device_name(device),
        "dtype": str(dtype),
        "num_iterations": num_iterations,
        "dim_size": dim_size,
        "num_matrices": num_matrices,
    }
    print(result)
    return result


def cartesian_product_dicts(d: Dict[Tuple[Any, ...], Any]) -> Tuple[Dict[Any, Any], ...]:
    return tuple(dict(zip(d, x)) for x in itertools.product(*d.values()))


inputs = cartesian_product_dicts(
    {
        "dim_size": (300,),
        "num_matrices": (6,),
        "dtype": (torch.float32, torch.float64),
        "device": (torch.device("cuda:0"), torch.device("cuda:1"), torch.device("cpu")),
        "func": (torch.matmul, logmatmulexp, logmatmulexp_lowmem),
        "num_iterations": (50,),
    }
)

json_path = os.path.expanduser(
    "~/projects/dctn/small_experiments/benchmark_logmatmulexp_results.json"
)

new_results: Tuple[Dict[str, Any], ...] = tuple(benchmark(**input) for input in inputs)
old_results: Tuple[Dict[str, Any], ...] = tuple(load_json(json_path)) if os.path.exists(
    json_path
) else ()
combined_results = old_results + new_results
save_json(combined_results, json_path)
