from functools import partial
from math import pi
from typing import *

import torch
from torch import Tensor
import torchvision.datasets
from torchvision.transforms.functional import to_tensor

from torch.utils.data import Dataset, DataLoader
from ignite.engine import Engine

φ_cos_sin_squared_1 = (lambda X: 2*(X*pi/2.).sin()**2, lambda X: 2*(X*pi/2.).cos()**2)

class MNISTLikeQuantumIndexedDataset(Dataset):
  def __init__(self, dataset_type: type, root: str, split: str,
               φ: Tuple[Callable[[Tensor], Tensor], ...]):
    if split == "train":
      torchvision_split = "train"
      torchvision_slice = slice(50000)
    elif split == "val":
      torchvision_split = "train"
      torchvision_slice = slice(50000, 60000)
    elif split == "test":
      torchvision_split = "test"
      torchvision_slice = slice(None)
    else:
      raise ValueError(f"{split=}")
    torchvision_dataset = dataset_type(root, torchvision_split, transform=to_tensor)
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
  φ: Tuple[Callable[[Tensor], Tensor], ...] = φ_cos_sin_squared_1
) -> Tuple[DataLoader, DataLoader, DataLoader]:
  """Returns train, val, and test dataloaders for `dataset_type`. Only train_dl shuffles."""
  assert dataset_type in (QuantumMNIST, QuantumFashionMNIST)
  train_ds, val_ds, test_ds = (dataset_type(root, s, φ) for s in ("train", "val", "test"))
  dl_partial = partial(DataLoader, batch_size=batch_size, collate_fn=collate_quantum,
                       pin_memory=(device.type == "cuda"),)
  train_dl = dl_partial(dataset=train_ds, shuffle=True, drop_last=True)
  val_dl, test_dl = (dl_partial(dataset=dataset) for dataset in (val_ds, test_ds))
  return train_dl, val_dl, test_dl
