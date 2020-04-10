import logging
from functools import partial
from math import pi
from typing import *

import torch
from torch import Tensor
import torchvision.datasets
from torchvision.transforms.functional import to_tensor

from torch.utils.data import Dataset, DataLoader
from ignite.engine import Engine

from .rank_one_tensor import RankOneTensorsBatch
from .align import align

φ_cos_sin_squared_1 = (lambda X: 2*(X*pi/2.).sin()**2, lambda X: 2*(X*pi/2.).cos()**2)

class MNISTLikeQuantumIndexedDataset(Dataset):
  def __init__(self, dataset_type: type, root: str, split: str,
               φ: Tuple[Callable[[Tensor], Tensor], ...]):
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
    x = torchvision_dataset.data[torchvision_slice].float() / 255. # shape: samples×h×w
    self.y = torchvision_dataset.targets[torchvision_slice] # shape: samples
    self.x = torch.stack(tuple(φ_i(x) for φ_i in φ), dim=3).unsqueeze(0)
    # self.x has shape: 1×samples×height×width×φ, where 1 is the number of channels

  def __len__(self) -> int:
    return len(self.y)

  def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
    return self.x[:, index], self.y[index], torch.tensor(index)


QuantumMNIST = partial(MNISTLikeQuantumIndexedDataset, torchvision.datasets.MNIST)
QuantumFashionMNIST = partial(MNISTLikeQuantumIndexedDataset, torchvision.datasets.FashionMNIST)

def collate_quantum(batch: List[Tuple[Tensor, Tensor, Tensor]]
) -> Tuple[Tensor, Tensor, Tensor]:
  x, y, indices = zip(*batch)
  return torch.stack(tuple(x), dim=1), torch.stack(tuple(y)), torch.stack(tuple(indices))


def get_data_loaders(dataset_type: type, root: str, batch_size: int, device: torch.device,
  φ: Tuple[Callable[[Tensor], Tensor], ...] = φ_cos_sin_squared_1,
  autoscale_kernel_size: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
  """Returns train, val, and test dataloaders for `dataset_type`. Only train_dl shuffles."""
  assert dataset_type in (QuantumMNIST, QuantumFashionMNIST)
  train_ds, val_ds, test_ds = (dataset_type(root, s, φ) for s in ("train", "val", "test"))
  if autoscale_kernel_size is not None:
    c = calc_scaling_factor(train_ds, autoscale_kernel_size, device)
    logging.debug(f"{c=}")
    train_ds.x *= c
    val_ds.x *= c
    test_ds.x *= c
    if (dataset_type is QuantumFashionMNIST and autoscale_kernel_size == 4
        and φ==φ_cos_sin_squared_1):
      assert torch.allclose(train_ds.x.mean(), torch.tensor(0.7284077405929565))
      assert torch.allclose(train_ds.x.std(), torch.tensor(0.6384438872337341))

  dl_partial = partial(DataLoader, batch_size=batch_size, collate_fn=collate_quantum,
                       pin_memory=(device.type == "cuda"),)
  train_dl = dl_partial(dataset=train_ds, shuffle=True, drop_last=True)
  val_dl, test_dl = (dl_partial(dataset=dataset) for dataset in (val_ds, test_ds))
  return train_dl, val_dl, test_dl


get_mnist_data_loaders = partial(get_data_loaders, QuantumMNIST)
get_fashionmnist_data_loaders = partial(get_data_loaders, QuantumFashionMNIST)

def calc_scaling_factor(ds: MNISTLikeQuantumIndexedDataset, kernel_size: int, device) -> float:
  """Calculates the number, by which `ds.x` must be multiplied in order to have its
  windows (of `kernel_size`) rank one tensors have μ^2+σ^2==1."""
  x = ds.x[:, :10880].to(device).double() # float32 mean and std works inaccurately
  x_windows = torch.cat(
    tuple(
      torch.stack(tuple(align(x_slice, kernel_size)), dim=0)
      for x_slice in x.split(128, dim=1)),
    dim=1)
  x_windows_r1t = RankOneTensorsBatch(x_windows, factors_dim=0, coordinates_dim=4)
  μ = x_windows_r1t.mean_over_batch().item()
  σ_squared = x_windows_r1t.var_over_batch().item()
  # I want to have μ^2+σ^2==1
  return (μ**2 + σ_squared)**(-1/(2*kernel_size**2))
