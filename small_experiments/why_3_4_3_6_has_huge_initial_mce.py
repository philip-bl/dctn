from os import environ
from typing import Tuple

environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

from dctn.eps import EPS
from dctn.eps_plus_linear import EPSesPlusLinear
from dctn.dataset_loading import QuantumFashionMNIST, get_data_loaders
from dctn.align import align
from dctn.rank_one_tensor import RankOneTensorsBatch
from dctn.epses_composition import make_epses_composition_unit_empirical_output_std

device = torch.device("cuda")
dtype = torch.float32
kernel_size_1 = 3
kernel_size_2 = 3
epses_specs = ((kernel_size_1, 4), (kernel_size_2, 6))
model = EPSesPlusLinear(epses_specs).to(device, dtype)

train_dl, val_dl, test_dl = get_data_loaders(
    QuantumFashionMNIST,
    "/mnt/hdd_1tb/datasets/fashionmnist",
    128,
    device,
    autoscale_kernel_size=3,
)

x0 = train_dl.dataset.x[:, :10880].to(device, dtype)
del train_dl, val_dl, test_dl

better_epses = make_epses_composition_unit_empirical_output_std(epses_specs, x0, device, dtype)
for eps_module, better_eps in zip(model.epses, better_epses):
    eps_module.core.data.copy_(better_eps)


def print_stats(x: torch.Tensor, representation_index: int) -> torch.Tensor:
    μ = x.mean()
    σ = x.std(unbiased=False)
    print(f"Rep {representation_index}: shape={x.shape}, {μ=:f}, {σ=:f}, {μ**2+σ**2=:f}")
    return σ


print_stats(x0, 0)


def make_windows(x: torch.Tensor, kernel_size: int) -> RankOneTensorsBatch:
    return RankOneTensorsBatch(
        torch.cat(
            tuple(
                torch.stack(tuple(align(x_slice, kernel_size)), dim=0)
                for x_slice in x.split(128, dim=1)
            ),
            dim=1,
        ),
        factors_dim=0,
        coordinates_dim=4,
    )


def print_windows_stats(x: torch.Tensor, kernel_size: int, representation_index: int):
    x_windows = make_windows(x, kernel_size)
    μ = x_windows.mean_over_batch()
    σ = x_windows.std_over_batch(unbiased=False)
    print(
        f"Rep {representation_index} windows: shape={x_windows.array.shape}, {μ=:f}, {σ=:f}, {μ**2+σ**2=:f}"
    )


print_windows_stats(x0, kernel_size_1, 0)


@torch.no_grad()
def apply_eps_in_slices(eps: EPS, x: torch.Tensor) -> torch.Tensor:
    return torch.cat(tuple(eps(x_slice) for x_slice in x.split(128, dim=1))).unsqueeze(0)


with torch.no_grad():
    x1 = apply_eps_in_slices(model.epses[0], x0)
σ1 = print_stats(x1, 1)
print_windows_stats(x1, kernel_size_2, 1)

with torch.no_grad():
    x2 = torch.cat(
        tuple(model.epses[1](x_slice) for x_slice in x1.split(128, dim=1))
    ).unsqueeze(0)
σ2 = print_stats(x2, 2)
torch.cuda.empty_cache()
# we have 1329 megabytes taken if float32, 1891 megabytes if float64

# ### Now, let me try to do rescaling of layers

# # rescale model.epses[0] - I want rep 1 to have σ=1.
# c1 = 1.0 / σ1
# model.epses[0].core.data.mul_(c1)


# # copy paste of diagnostic code above
# with torch.no_grad():
#     x1 = apply_eps_in_slices(model.epses[0], x0)
# σ1 = print_stats(x1, 1)
# print_windows_stats(x1, kernel_size_2, 1)

# with torch.no_grad():
#     x2 = torch.cat(
#         tuple(model.epses[1](x_slice) for x_slice in x1.split(128, dim=1))
#     ).unsqueeze(0)
# σ2 = print_stats(x2, 2)
# torch.cuda.empty_cache()

# # rescale model.epses[1] - I want rep 2 to have σ=1.
# c2 = 1.0 / σ2
# model.epses[1].core.data.mul_(c2)

# # copy paste of diagnostic code above
# with torch.no_grad():
#     x1 = apply_eps_in_slices(model.epses[0], x0)
# σ1 = print_stats(x1, 1)
# print_windows_stats(x1, kernel_size_2, 1)

# with torch.no_grad():
#     x2 = torch.cat(
#         tuple(model.epses[1](x_slice) for x_slice in x1.split(128, dim=1))
#     ).unsqueeze(0)
# σ2 = print_stats(x2, 2)
# torch.cuda.empty_cache()
