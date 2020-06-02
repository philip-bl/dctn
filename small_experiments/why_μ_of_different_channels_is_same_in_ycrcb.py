import torch

from dctn.dataset_loading import CIFAR10ColoredIndexedDataset

ds = CIFAR10ColoredIndexedDataset("YCbCr", "/mnt/hdd_1tb/datasets/cifar10", "train")

print(ds.x.shape)

print(ds.x.mean(dim=(0, 1, 2, 3)))

x_flattened = ds.x.reshape(-1, 3)

print(x_flattened.mean(dim=0))
print(f"{x_flattened[:, 0].mean()=}, {x_flattened[:, 0].mean()=}, {x_flattened[:, 0].mean()=}")
print(x_flattened.double().mean(dim=0))


# WEIRD, let's look at the original torchvision CIFAR10 dataset

from typing import Tuple
from PIL.Image import Image
from einops import rearrange
from torchvision.datasets import CIFAR10
from torchvision.transforms.functional import to_tensor

tv_ds = CIFAR10("/mnt/hdd_1tb/datasets/cifar10")

YCbCr_images: Tuple[Image, ...] = tuple(image.convert("YCbCr") for image, label in tv_ds)
x = torch.stack([to_tensor(image) for image in YCbCr_images])
print(x.shape)  # (50000, 3, 32, 32)
print(x.mean(dim=(0, 2, 3)))
x_rearranged = rearrange(x, "b c h w -> b h w c")
x_rearranged.mean(dim=(0, 1, 2))

# WTF WTF WTF!!!!!!!!


# Let's try to replicate what is happening in the first part but simpler here

from typing import Tuple
from PIL.Image import Image
from einops import rearrange
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms.functional import to_tensor

tv_ds = CIFAR10("/mnt/hdd_1tb/datasets/cifar10")
YCbCr_images: Tuple[Image, ...] = tuple(image.convert("YCbCr") for image, label in tv_ds)
tensors: Tuple[torch.Tensor, ...] = tuple(
    rearrange(to_tensor(image), "c h w -> h w c") for image in YCbCr_images
)
x = torch.stack(tensors).unsqueeze(0)  # 1 × 50000 × 32 × 32 × 3
x_flattened = x.reshape(-1, 3)
print(x_flattened.mean(dim=0))  # (0.3277, 0.3277, 0.3277)
for channel in range(3):
    print(x_flattened[:, channel].mean())  # (0.4794, 0.4916, 0.5037)
print(x_flattened.double().mean(dim=0))  # (0.4790, 0.4809, 0.5077)
print(x_flattened.cuda().mean(dim=0))  # (0.4790, 0.4809, 0.5077)
