from random import seed
import os
from logging import getLogger

from typing import List, Tuple, Optional, Callable, Any, Dict

from libcrap import shuffled

import numpy as np

from PIL.Image import Image

from attr import attrib, attrs

import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose
from torchvision.transforms.functional import to_grayscale, to_tensor, resize, to_pil_image


CIFAR10_NUM_TRAIN_SAMPLES = 45000


def _to_28x28_grayscale_tensor(
    ds: CIFAR10, save_examples_to_dir: Optional[str] = None
) -> torch.FloatTensor:
    x_pil_images: Tuple[Image] = tuple(to_pil_image(image) for image in ds.data)
    x_28x28: Tuple[Image] = tuple(resize(image, (28, 28)) for image in x_pil_images)
    x_28x28_grayscale: Tuple[Image] = tuple(to_grayscale(image) for image in x_28x28)

    if save_examples_to_dir is not None:
        for i in range(20):
            x_28x28_grayscale[i].save(
                os.path.join(
                    save_examples_to_dir,
                    f"28x28_grayscale_{i}_{ds.classes[ds.targets[i]]}.png",
                )
            )
    return torch.cat(
        tuple(to_tensor(image) for image in x_28x28_grayscale)
    )  # (N, 28, 28); ∈ [0., 1.]


@attrs(auto_attribs=True, frozen=True)
class DatasetAsTensors:
    x_train: torch.FloatTensor
    y_train: torch.LongTensor
    indices_train: torch.LongTensor

    x_val: torch.FloatTensor
    y_val: torch.LongTensor
    indices_val: torch.LongTensor

    x_test: torch.FloatTensor
    y_test: torch.LongTensor
    indices_test: torch.LongTensor


def load_cifar10_as_grayscale_tensors(ds_path: str) -> DatasetAsTensors:
    ds_train_and_val = CIFAR10(ds_path, train=True)
    x: np.ndarray = ds_train_and_val.data  # (50000, 32, 32, 3)
    y: List[int] = ds_train_and_val.targets
    x_tensor = _to_28x28_grayscale_tensor(ds_train_and_val, ds_path)

    # shuffle the training dataset
    seed(0)
    shuffled_indices: List[int] = shuffled(range(len(x)))
    getLogger(f"{__name__}.{load_cifar10_as_grayscale_tensors.__qualname__}").info(
        f"{hash(tuple(shuffled_indices))=}, {shuffled_indices[:10]=}"
    )
    # 6271394816323448769 and (25247, 49673, 27562, 2653, 16968, 33506, 31845, 26537, 19877, 31234)

    x_tensor_shuffled: torch.Tensor = x_tensor[shuffled_indices]
    y_shuffled: List[int] = np.array(y)[shuffled_indices].tolist()

    return DatasetAsTensors(
        x_train=x_tensor_shuffled[:CIFAR10_NUM_TRAIN_SAMPLES],  # (45000, 28, 28)
        y_train=torch.tensor(y_shuffled[:CIFAR10_NUM_TRAIN_SAMPLES]),
        indices_train=torch.tensor(shuffled_indices[:CIFAR10_NUM_TRAIN_SAMPLES]),
        x_val=x_tensor_shuffled[CIFAR10_NUM_TRAIN_SAMPLES:],
        y_val=torch.tensor(y_shuffled[CIFAR10_NUM_TRAIN_SAMPLES:]),
        indices_val=torch.tensor(shuffled_indices[CIFAR10_NUM_TRAIN_SAMPLES:]),
        x_test=_to_28x28_grayscale_tensor(ds_test := CIFAR10(ds_path, train=False)),
        y_test=torch.tensor(ds_test.targets),
        indices_test=torch.tensor(range(len(ds_test))),
    )
