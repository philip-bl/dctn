import sys
import os
import random
from itertools import chain

import numpy as np

import torch

import opt_einsum as oe

import einops

sys.path.insert(0, os.path.expanduser("~/projects/torchmps"))
from torchmps import MPS

# This file uses commit adb993e of my fork of TorchMPS

torch.set_default_dtype(torch.float64)

init_std = 1e-2
print(f"{init_std=:.1e}")
mpo = MPS(
    input_dim=9, output_dim=10, bond_dim=11, init_std=init_std, adaptive_mode=True
).train(False)


my_cores = (
    *(mpo.init_input1_tensor[c] for c in range(4)),  # l×r×in
    mpo.init_output_tensor,  # o×l×r
    *(mpo.init_input2_tensor[c] for c in range(5)),  # l×r×in
)
my_merged_cores = (
    # here b{i} means bond number i
    oe.contract(
        my_cores[0],
        ("b0", "b1", "in0"),
        my_cores[1],
        ("b1", "b2", "in1"),
        ("b0", "b2", "in0", "in1"),
    ),
    oe.contract(
        my_cores[2],
        ("b2", "b3", "in2"),
        my_cores[3],
        ("b3", "b4", "in3"),
        ("b2", "b4", "in2", "in3"),
    ),
    oe.contract(
        my_cores[4],
        ("out", "b4", "b5"),
        my_cores[5],
        ("b5", "b6", "in4"),
        ("out", "b4", "b6", "in4"),
    ),
    oe.contract(
        my_cores[6],
        ("b6", "b7", "in5"),
        my_cores[7],
        ("b7", "b8", "in6"),
        ("b6", "b8", "in5", "in6"),
    ),
    oe.contract(
        my_cores[8],
        ("b8", "b9", "in7"),
        my_cores[9],
        ("b9", "b10", "in8"),
        ("b8", "b10", "in7", "in8"),
    ),
)
assert torch.allclose(my_merged_cores[0], mpo.linear_region.module_list[0].tensor[0])
assert torch.allclose(my_merged_cores[1], mpo.linear_region.module_list[0].tensor[1])
assert torch.allclose(my_merged_cores[2], mpo.linear_region.module_list[1].tensor)
assert torch.allclose(my_merged_cores[3], mpo.linear_region.module_list[2].tensor[0])
assert torch.allclose(my_merged_cores[4], mpo.linear_region.module_list[2].tensor[1])


# Now, for each (merged) core of the MPO, I will contract it with
# its two input vectors
input = torch.randn(1, 9, 2) + 0.3
processed_cores = (
    *mpo.linear_region.module_list[0](input[:, :4])
    .tensor.squeeze(0)
    .split(1),  # batch × lbond × rbond
    mpo.linear_region.module_list[1](input[:, 4]).tensor,  # batch × out × lbond × rbond
    *mpo.linear_region.module_list[2](input[:, 5:])
    .tensor.squeeze(0)
    .split(1),  # batch × lbond × rbond
)
my_processed_cores = (
    oe.contract(
        my_merged_cores[0],
        ("b0", "b2", "in0", "in1"),
        input[:, 0],
        ("batch", "in0"),
        input[:, 1],
        ("batch", "in1"),
        ("batch", "b0", "b2"),
    ),
    oe.contract(
        my_merged_cores[1],
        ("b2", "b4", "in2", "in3"),
        input[:, 2],
        ("batch", "in2"),
        input[:, 3],
        ("batch", "in3"),
        ("batch", "b2", "b4"),
    ),
    oe.contract(
        my_merged_cores[2],
        ("out", "b4", "b6", "in4"),
        input[:, 4],
        ("batch", "in4"),
        ("batch", "out", "b4", "b6"),
    ),
    oe.contract(
        my_merged_cores[3],
        ("b6", "b8", "in5", "in6"),
        input[:, 5],
        ("batch", "in5"),
        input[:, 6],
        ("batch", "in6"),
        ("batch", "b6", "b8"),
    ),
    oe.contract(
        my_merged_cores[4],
        ("b8", "b10", "in7", "in8"),
        input[:, 7],
        ("batch", "in7"),
        input[:, 8],
        ("batch", "in8"),
        ("batch", "b8", "b10"),
    ),
)
for (core, my_core) in zip(processed_cores, my_processed_cores):
    assert torch.allclose(my_core, core)


# now I check if the whole result is correct
output = mpo(input)
my_output = oe.contract(
    torch.tensor([1.0] + [0.0] * 10),
    ("b0",),
    my_processed_cores[0],
    ("batch", "b0", "b2"),
    my_processed_cores[1],
    ("batch", "b2", "b4"),
    my_processed_cores[2],
    ("batch", "out", "b4", "b6"),
    my_processed_cores[3],
    ("batch", "b6", "b8"),
    my_processed_cores[4],
    ("batch", "b8", "b10"),
    torch.tensor([1.0] + [0.0] * 10),
    ("b10",),
    ("batch", "out"),
)
assert torch.allclose(my_output, output)


# now I check the properties of the matrix equivalent to the MPO
explicit_tensor = oe.contract(
    torch.tensor([1.0] + [0.0] * 10),
    ("b0",),
    my_merged_cores[0],
    ("b0", "b2", "in0", "in1"),
    my_merged_cores[1],
    ("b2", "b4", "in2", "in3"),
    my_merged_cores[2],
    ("out", "b4", "b6", "in4"),
    my_merged_cores[3],
    ("b6", "b8", "in5", "in6"),
    my_merged_cores[4],
    ("b8", "b10", "in7", "in8"),
    torch.tensor([1.0] + [0.0] * 10),
    ("b10",),
    ("out", "in0", "in1", "in2", "in3", "in4", "in5", "in6", "in7", "in8"),
)
# check that it implements the same function
explicit_tensor_output = oe.contract(
    explicit_tensor,
    ("out", "in0", "in1", "in2", "in3", "in4", "in5", "in6", "in7", "in8"),
    *chain.from_iterable((input[:, c], ("batch", f"in{c}")) for c in range(9)),
)
assert torch.allclose(explicit_tensor_output, output)
matrix = einops.rearrange(
    explicit_tensor,
    "out in0 in1 in2 in3 in4 in5 in6 in7 in8 -> out (in0 in1 in2 in3 in4 in5 in6 in7 in8)",
).numpy()
print(f"{np.linalg.cond(matrix)=:.1e}")
print(f"{matrix.mean()=:.1e}, {matrix.std()=:.1e}")
# for init_std=1e-1, cond is between 70  and 200,
#          mean is between 0.7   and 1.5
#          std  is between 0.1   and 0.5
# for init_std=1e-2, cond is between 30000 and 150000
#          mean is between 0.98  and 1
#          std  is between 1e-2  and 3e-2
# for init_std=1e-3, cond is between 7e6   and 2e8
