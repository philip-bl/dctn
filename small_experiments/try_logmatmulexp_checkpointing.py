from functools import reduce

import torch
import torch.cuda

from dctn.logmatmulexp import logmatmulexp_lowmem, logmatmulexp


def human_readable_size(size, decimal_places=0):
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if size < 1024.0:
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f}{unit}"


device = torch.device("cuda")
dtype = torch.float64
bytes_per_number = torch.finfo(dtype).bits // 8

# for func, choose one of logmatmulexp, logmatmulexp_lowmem, torch.matmul
func = logmatmulexp_lowmem


def print_allocations_info() -> None:
    stats = torch.cuda.memory_stats(device)
    bytes_currently = stats["allocated_bytes.all.current"]
    bytes_historically = stats["allocated_bytes.all.allocated"]
    bytes_peak = stats["allocated_bytes.all.peak"]
    numbers_currently = bytes_currently / bytes_per_number
    numbers_historically = bytes_historically / bytes_per_number
    numbers_peak = bytes_peak / bytes_per_number
    print(
        f"""Approximately this many bytes and {dtype} numbers have been allocated:
Currently: {human_readable_size(bytes_currently)} = {numbers_currently} numbers
Historically: {human_readable_size(bytes_historically)} = {numbers_historically} numbers
Peak: {human_readable_size(bytes_peak)} = {numbers_peak} numbers"""
    )


print("Before creating matrices\n")
print_allocations_info()

log_matrices = tuple(torch.randn(100, 100, dtype=dtype, device=device) for i in range(20))

print("\nAfter creating matrices\n")
print_allocations_info()

log_matrices[0].requires_grad_()
print("\nAfter setting requires_grad=True on the first matrix\n")
print_allocations_info()

print(f"\nUsing {func.__qualname__}")
result = reduce(func, log_matrices)
print("\nAfter forward pass\n")
print_allocations_info()

result.backward(torch.ones_like(result))
print("\nAfter backward pass\n")
print_allocations_info()
