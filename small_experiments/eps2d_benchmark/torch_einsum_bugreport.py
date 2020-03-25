"""https://github.com/pytorch/pytorch/issues/35299"""

import torch
big_core = torch.randn(
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    dtype=torch.float64, device="cuda", requires_grad=True
)
batches_of_small_cores = [
    torch.randn(8, 25, 25, 2, dtype=torch.float64, device="cuda", requires_grad=True)
    for _ in range(16)
]

result = torch.einsum(
    'qrsp,qrso,qrsn,qrsm,qrsl,qrsk,qrsj,qrsi,qrsh,qrsg,qrsf,qrse,qrsd,qrsc,qrsb,qrsa,abcdefghijklmnopt->qrst', *batches_of_small_cores, big_core
)

peak_allocated_GiB = torch.cuda.memory_stats()["allocated_bytes.all.peak"] / 1024 / 1024 / 1024
print(peak_allocated_GiB)
total_allocated_GiB = torch.cuda.memory_stats()["allocated_bytes.all.allocated"] / 1024 / 1024 / 1024
print(total_allocated_GiB)
# peak allocated GiB: 4.9
# total allocated GiB: 4.9



import torch
big_core = torch.randn(
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    dtype=torch.float64, device="cuda", requires_grad=True
)
batches_of_small_cores = [
    torch.randn(8, 25, 25, 2, dtype=torch.float64, device="cuda", requires_grad=True)
    for _ in range(16)
]

result = torch.einsum(
    'abcdefghijklmnopt,qrsp,qrso,qrsn,qrsm,qrsl,qrsk,qrsj,qrsi,qrsh,qrsg,qrsf,qrse,qrsd,qrsc,qrsb,qrsa->qrst', big_core, *batches_of_small_cores
)


peak_allocated_GiB = torch.cuda.memory_stats()["allocated_bytes.all.peak"] / 1024 / 1024 / 1024
print(peak_allocated_GiB)
total_allocated_GiB = torch.cuda.memory_stats()["allocated_bytes.all.allocated"] / 1024 / 1024 / 1024
print(total_allocated_GiB)
# peak allocated GiB: 6.1
# total allocated GiB: 7.3

