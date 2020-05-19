from random import seed
import os
from logging import getLogger
from functools import partial
from math import pi
from typing import Tuple, Callable, List, Optional, Dict, Any

from libcrap import shuffled


import numpy as np

from PIL.Image import Image

from attr import attrib, attrs

from einops import rearrange

import torch
from torch import Tensor
import torchvision.datasets
from torchvision.transforms import Compose
from torchvision.transforms.functional import to_grayscale, to_tensor, resize, to_pil_image


from torch.utils.data import Dataset, DataLoader
from ignite.engine import Engine

from .rank_one_tensor import RankOneTensorsBatch
from .align import align
from .utils import implies, xor

φ_cos_sin_squared_1 = (
    lambda X: 2 * (X * pi / 2.0).sin() ** 2,
    lambda X: 2 * (X * pi / 2.0).cos() ** 2,
)


class MNISTLikeQuantumIndexedDataset(Dataset):
    def __init__(
        self,
        dataset_type: type,
        root: str,
        split: str,
        φ: Tuple[Callable[[Tensor], Tensor], ...],
    ):
        if split == "train":
            torchvision_train = True
            torchvision_slice = slice(50000)
        elif split == "val":
            torchvision_train = True
            torchvision_slice = slice(50000, 60000)
        elif split == "test":
            torchvision_train = False
            torchvision_slice = slice(None)
        else:
            raise ValueError(f"{split=}")
        torchvision_dataset = dataset_type(root, train=torchvision_train, transform=to_tensor)
        self.unmodified_x = (
            torchvision_dataset.data[torchvision_slice].float() / 255.0
        )  # shape: samples×h×w
        self.y = torchvision_dataset.targets[torchvision_slice]  # shape: samples
        self.x = torch.stack(tuple(φ_i(self.unmodified_x) for φ_i in φ), dim=3).unsqueeze(0)
        # self.x has shape: 1×samples×height×width×φ, where 1 is the number of channels

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        return self.x[:, index], self.y[index], torch.tensor(index)


QuantumMNIST = partial(MNISTLikeQuantumIndexedDataset, torchvision.datasets.MNIST)
QuantumFashionMNIST = partial(
    MNISTLikeQuantumIndexedDataset, torchvision.datasets.FashionMNIST
)


def calc_scaling_factor(ds: MNISTLikeQuantumIndexedDataset, kernel_size: int, device) -> float:
    """Calculates the number, by which `ds.x` must be multiplied in order to have its
  windows (of `kernel_size`) rank one tensors have μ^2+σ^2==1."""
    x = ds.x[:, :10880].to(device).double()  # float32 mean and std works inaccurately
    x_windows = torch.cat(
        tuple(
            torch.stack(tuple(align(x_slice, kernel_size)), dim=0)
            for x_slice in x.split(128, dim=1)
        ),
        dim=1,
    )
    x_windows_r1t = RankOneTensorsBatch(x_windows, factors_dim=0, coordinates_dim=4)
    μ = x_windows_r1t.mean_over_batch().item()
    σ_squared = x_windows_r1t.var_over_batch().item()
    # I want to have μ^2+σ^2==1
    return (μ ** 2 + σ_squared) ** (-1 / (2 * kernel_size ** 2))


CIFAR10_NUM_TRAIN_SAMPLES = 45000


def _to_28x28_grayscale_tensor(
    ds: torchvision.datasets.CIFAR10, save_examples_to_dir: Optional[str] = None
) -> torch.FloatTensor:
    """The returned tensor has shape (dataset_size, 28, 28),
    and all components are in range [0. 1.]"""
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
    return torch.cat(tuple(to_tensor(image) for image in x_28x28_grayscale))


def _to_32x32_grayscale_tensor(ds: torchvision.datasets.CIFAR10) -> torch.FloatTensor:
    """The returned tensor has shape (dataset_size, 32, 32),
    and all components are in range [0. 1.]"""
    x_pil_images: Tuple[Image] = tuple(to_pil_image(image) for image in ds.data)
    x_grayscale: Tuple[Image] = tuple(to_grayscale(image) for image in x_pil_images)
    return torch.cat(tuple(to_tensor(image) for image in x_grayscale))


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


def load_cifar10_as_grayscale_tensors(ds_path: str, image_size: int) -> DatasetAsTensors:
    assert image_size in (28, 32)
    ds_train_and_val = torchvision.datasets.CIFAR10(ds_path, train=True)
    x: np.ndarray = ds_train_and_val.data  # (50000, 32, 32, 3)
    y: List[int] = ds_train_and_val.targets
    x_tensor = (
        _to_28x28_grayscale_tensor(ds_train_and_val)
        if image_size == 28
        else _to_32x32_grayscale_tensor(ds_train_and_val)
    )

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
        x_train=x_tensor_shuffled[:CIFAR10_NUM_TRAIN_SAMPLES],  # (45000, height, width)
        y_train=torch.tensor(y_shuffled[:CIFAR10_NUM_TRAIN_SAMPLES]),
        indices_train=torch.tensor(shuffled_indices[:CIFAR10_NUM_TRAIN_SAMPLES]),
        x_val=x_tensor_shuffled[CIFAR10_NUM_TRAIN_SAMPLES:],
        y_val=torch.tensor(y_shuffled[CIFAR10_NUM_TRAIN_SAMPLES:]),
        indices_val=torch.tensor(shuffled_indices[CIFAR10_NUM_TRAIN_SAMPLES:]),
        x_test=(
            _to_28x28_grayscale_tensor if image_size == 28 else _to_32x32_grayscale_tensor
        )(ds_test := torchvision.datasets.CIFAR10(ds_path, train=False)),
        y_test=torch.tensor(ds_test.targets),
        indices_test=torch.tensor(range(len(ds_test))),
    )


def load_cifar10_as_colored_tensors(ds_path: str, colors: str) -> DatasetAsTensors:
    assert colors in ("rgb", "YCbCr")
    ds_train_and_val = torchvision.datasets.CIFAR10(ds_path, train=True)
    ds_test = torchvision.datasets.CIFAR10(ds_path, train=False)
    images_train: Tuple[Image, ...] = tuple(
        ds_train_and_val[i][0] for i in range(len(ds_train_and_val))
    )
    images_test: Tuple[Image, ...] = tuple(ds_test[i][0] for i in range(len(ds_test)))
    if colors == "YCbCr":
        images_train = tuple(image.convert("YCbCr") for image in images_train)
        images_test = tuple(image.convert("YCbCr") for image in images_test)

    x: torch.FloatTensor = torch.stack(
        [rearrange(to_tensor(image), "c h w -> h w c") for image in images_train]
    ).unsqueeze(
        0
    )  # 1 × 50000 × 32 × 32 × 3
    y: List[int] = ds_train_and_val.targets

    x_test: torch.FloatTensor = torch.stack(
        [rearrange(to_tensor(image), "c h w -> h w c") for image in images_test]
    ).unsqueeze(
        0
    )  # 1 × 10000 × 32 × 32 × 3

    # shuffle the training dataset
    seed(0)
    shuffled_indices: List[int] = shuffled(range(len(y)))
    getLogger(f"{__name__}.{load_cifar10_as_colored_tensors.__qualname__}").info(
        f"{hash(tuple(shuffled_indices))=}, {shuffled_indices[:10]=}"
    )
    # 6271394816323448769 and (25247, 49673, 27562, 2653, 16968, 33506, 31845, 26537, 19877, 31234)

    x_shuffled: torch.Tensor = x[:, shuffled_indices]
    y_shuffled: List[int] = np.array(y)[shuffled_indices].tolist()

    return DatasetAsTensors(
        x_train=x_shuffled[:, :CIFAR10_NUM_TRAIN_SAMPLES],  # (1, 45000, height, width)
        y_train=torch.tensor(y_shuffled[:CIFAR10_NUM_TRAIN_SAMPLES]),
        indices_train=torch.tensor(shuffled_indices[:CIFAR10_NUM_TRAIN_SAMPLES]),
        x_val=x_shuffled[:, CIFAR10_NUM_TRAIN_SAMPLES:],
        y_val=torch.tensor(y_shuffled[CIFAR10_NUM_TRAIN_SAMPLES:]),
        indices_val=torch.tensor(shuffled_indices[CIFAR10_NUM_TRAIN_SAMPLES:]),
        x_test=x_test,
        y_test=torch.tensor(ds_test.targets),
        indices_test=torch.tensor(range(len(ds_test))),
    )


class _CIFAR10GrayscaleQuantumIndexedDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        φ: Tuple[Callable[[torch.FloatTensor], torch.FloatTensor], ...],
        image_size: int,
    ):
        tensors = load_cifar10_as_grayscale_tensors(root, image_size)
        self.unmodified_x = getattr(
            tensors, f"x_{split}"
        )  # actually modified. samples × h × w
        self.y = getattr(tensors, f"y_{split}")  # shape: samples
        self.x = torch.stack(tuple(φ_i(self.unmodified_x) for φ_i in φ), dim=3).unsqueeze(0)
        # self. x has shape 1 × samples × height × width × φ, where 1 is the number of channels
        self.indices = getattr(tensors, f"indices_{split}")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(
        self, i: int
    ) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor]:
        return self.x[:, i], self.y[i], self.indices[i]


class CIFAR10ColoredIndexedDataset(Dataset):
    def __init__(self, colors: str, root: str, split: str):
        assert colors in ("rgb", "YCbCr")
        tensors = load_cifar10_as_colored_tensors(root, colors)
        self.y = getattr(tensors, f"y_{split}")
        self.x = getattr(tensors, f"x_{split}")
        self.indices = getattr(tensors, f"indices_{split}")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(
        self, i: int
    ) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor]:
        return self.x[:, i], self.y[i], self.indices[i]


class CIFAR1028x28GrayscaleQuantumIndexedDataset(_CIFAR10GrayscaleQuantumIndexedDataset):
    def __init__(self, *args):
        super().__init__(*args, image_size=28)


class CIFAR1032x32GrayscaleQuantumIndexedDataset(_CIFAR10GrayscaleQuantumIndexedDataset):
    def __init__(self, *args):
        super().__init__(*args, image_size=32)


def collate_quantum(
    batch: List[Tuple[Tensor, Tensor, Tensor]]
) -> Tuple[Tensor, Tensor, Tensor]:
    x, y, indices = zip(*batch)
    return torch.stack(tuple(x), dim=1), torch.stack(tuple(y)), torch.stack(tuple(indices))


def get_data_loaders(
    dataset_type: type,
    root: str,
    batch_size: int,
    device: torch.device,
    φ: Tuple[Callable[[Tensor], Tensor], ...] = φ_cos_sin_squared_1,
    autoscale_kernel_size: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Returns train, val, and test dataloaders for `dataset_type`. Only train_dl shuffles."""
    assert dataset_type in (
        QuantumMNIST,
        QuantumFashionMNIST,
        CIFAR1028x28GrayscaleQuantumIndexedDataset,
        CIFAR1032x32GrayscaleQuantumIndexedDataset,
    )
    train_ds, val_ds, test_ds = (dataset_type(root, s, φ) for s in ("train", "val", "test"))
    if autoscale_kernel_size is not None:
        ν = calc_scaling_factor(train_ds, autoscale_kernel_size, device)
        getLogger(f"{__name__}.{get_data_loaders.__qualname__}").info(f"{ν=}")
        train_ds.x *= ν
        val_ds.x *= ν
        test_ds.x *= ν
        if (
            dataset_type is QuantumFashionMNIST
            and autoscale_kernel_size == 4
            and φ == φ_cos_sin_squared_1
        ):
            assert torch.allclose(train_ds.x.mean(), torch.tensor(0.7284077405929565))
            assert torch.allclose(train_ds.x.std(), torch.tensor(0.6384438872337341))

    dl_partial = partial(
        DataLoader,
        batch_size=batch_size,
        collate_fn=collate_quantum,
        pin_memory=(device.type == "cuda"),
    )
    train_dl = dl_partial(dataset=train_ds, shuffle=True, drop_last=True)
    val_dl, test_dl = (dl_partial(dataset=dataset) for dataset in (val_ds, test_ds))
    torch.cuda.empty_cache()
    return train_dl, val_dl, test_dl


def get_cifar10_colored_data_loaders(
    colors: str,
    root: str,
    batch_size: int,
    device: torch.device,
    center_and_normalize_each_channel: bool,
    add_constant_channel: Optional[float] = None,
    ν: Optional[Tuple[float, float, float]] = None,
    autoscale_kernel_size: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Returns train_ds, val_ds, and test_ds for CIFAR10 32x32 with Q_0=3. Only `train_ds` shuffles."""
    assert colors in ("rgb", "YCbCr")
    assert xor(autoscale_kernel_size is not None, ν is not None)
    dses = tuple(
        CIFAR10ColoredIndexedDataset(colors, root, split) for split in ("train", "val", "test")
    )
    train_ds, val_ds, test_ds = dses
    logger = getLogger(f"{__name__}.{get_cifar10_colored_data_loaders.__qualname__}")
    if center_and_normalize_each_channel:
        assert train_ds.x.shape == (1, 45000, 32, 32, 3)
        μ = train_ds.x.double().mean(dim=(0, 1, 2, 3))
        σ = train_ds.x.double().std(dim=(0, 1, 2, 3), unbiased=False)
        for ds in (train_ds, val_ds, test_ds):
            ds.x -= μ
            ds.x /= σ
        logger.info(
            f"In training datasets images' channels had μ={μ.tolist()}, σ={σ.tolist()}, normalized them to zero μ and unit σ"
        )
    if add_constant_channel is not None:
        for ds in (train_ds, val_ds, test_ds):
            # ds.x has shape 1 × ds_size × 32 × 32 × 3
            ds.x = torch.cat(
                (ds.x, add_constant_channel * torch.ones_like(ds.x[:, :, :, :, :1])), dim=4
            )
        logger.info(
            f"After (maybe) centering and normalization, but before (maybe) autoscaling based on kernel size, and multiplying by ν, added a channel of with all values {add_constant_channel:.30e}"
        )
        if ν is not None:
            ν = ν + (1.0,)
    if autoscale_kernel_size is not None:
        ν_single_value: float = calc_scaling_factor(train_ds, autoscale_kernel_size, device)
        ν = (ν_single_value, ν_single_value, ν_single_value, ν_single_value)
        logger.info(f"calc_scaling_factor chose {ν=}")
    for ds in dses:
        ds.x *= torch.tensor(ν)
    logger.info(
        f"After multiplying by {ν=}, train_ds.x.σ={train_ds.x.std(dim=(0,1,2,3), unbiased=False).tolist()}"
    )
    dl_partial = partial(
        DataLoader,
        batch_size=batch_size,
        collate_fn=collate_quantum,
        pin_memory=(device.type == "cuda"),
    )
    train_dl = dl_partial(dataset=train_ds, shuffle=True, drop_last=True)
    val_dl = dl_partial(dataset=val_ds)
    test_dl = dl_partial(dataset=test_ds)
    torch.cuda.empty_cache()
    return train_dl, val_dl, test_dl


get_mnist_data_loaders = partial(get_data_loaders, QuantumMNIST)
get_fashionmnist_data_loaders = partial(get_data_loaders, QuantumFashionMNIST)
get_cifar10_28x28_grayscale_data_loaders = partial(
    get_data_loaders, CIFAR1028x28GrayscaleQuantumIndexedDataset
)
get_cifar10_32x32_grayscale_data_loaders = partial(
    get_data_loaders, CIFAR1032x32GrayscaleQuantumIndexedDataset
)
