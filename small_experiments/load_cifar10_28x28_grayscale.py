from random import seed
import os
from logging import getLogger

from typing import List, Tuple, Optional, Callable, Any

from libcrap import shuffled

import numpy as np

from PIL.Image import Image

import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose
from torchvision.transforms.functional import to_grayscale, to_tensor, resize, to_pil_image


ds_path = "/mnt/hdd_1tb/datasets/cifar10"
num_train_samples = 45000
num_val_samples = 5000


ds_train_and_val = CIFAR10(ds_path, train=True)
x: np.ndarray = ds_train_and_val.data  # (50000, 32, 32, 3)
assert len(x) == num_train_samples + num_val_samples
y: List[int] = ds_train_and_val.targets


def to_28x28_grayscale_tensor(
    ds: CIFAR10, save_examples_to_dir: Optional[str] = None
) -> torch.FloatTensor:
    x_pil_images: Tuple[Image] = tuple(to_pil_image(image) for image in ds.data)
    x_28x28: Tuple[Image] = tuple(resize(image, (28, 28)) for image in x_pil_images)
    x_28x28_grayscale: Tuple[Image] = tuple(to_grayscale(image) for image in x_28x28)

    if save_examples_to_dir is not None:
        for i in range(20):
            x_28x28_grayscale[i].save(
                os.path.join(
                    save_examples_to_dir, f"28x28_grayscale_{i}_{ds.classes[y[i]]}.png"
                )
            )
    return torch.cat(
        tuple(to_tensor(image) for image in x_28x28_grayscale)
    )  # (N, 28, 28); ∈ [0., 1.]


x_tensor = to_28x28_grayscale_tensor(ds_train_and_val, ds_path)

# shuffle the training dataset
seed(0)
shuffled_indices: List[int] = shuffled(range(len(x)))
getLogger(__name__).info(f"{hash(tuple(shuffled_indices))=}, {shuffled_indices[:10]=}")
# 6271394816323448769 and (25247, 49673, 27562, 2653, 16968, 33506, 31845, 26537, 19877, 31234)

x_tensor_shuffled: torch.Tensor = x_tensor[shuffled_indices]
y_shuffled: List[int] = np.array(y)[shuffled_indices].tolist()

x_train: torch.FloatTensor = x_tensor_shuffled[:num_train_samples]  # (45000, 28, 28)
y_train: torch.LongTensor = torch.tensor(y_shuffled[:num_train_samples])

x_val: torch.FloatTensor = x_tensor_shuffled[num_train_samples:]
y_val: torch.LongTensor = torch.tensor(y_shuffled[num_train_samples:])

ds_test = CIFAR10(ds_path, train=False)
x_test: torch.FloatTensor = to_28x28_grayscale_tensor(ds_test)
y_test: torch.LongTensor = torch.tensor(ds_test.targets)
