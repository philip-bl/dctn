from functools import partial

import torch
from torch import Tensor
import torchvision.datasets
from torchvision.transforms.functional import to_tensor

from torch.utils.data import Dataset
from ignite.engine import Engine

φ_bearlamp = (lambda X: 2*X.sin()**2, lambda X: 2*X.cos()**2)

class MNISTLikeQuantumIndexedDataset(Dataset):
  def __init__(self, dataset_type: type, root: str, split: str = "train", φ=φ_bearlamp):
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
    X = torchvision_dataset.data[torchvision_slice].float() / 255. # shape: samples×h×w
    self.y = torchvision_dataset.targets[torchvision_slice] # shape: samples
    self.X = torch.cat((φ[0](X).unsqueeze(3), φ[1](X).unsqueeze(3)), dim=3) \
      .unsqueeze(0) # shape: 1×samples×height×width×2, where 1 is the number of channels

  def __len__(self) -> int:
    return len(self.y)

  def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
    return self.X[:, index], self.y[index], torch.tensor(index)


QuantumMnist = partial(MNISTLikeQuantumIndexedDataset, torchvision.datasets.MNIST)
QuantumFashionMNIST = partial(MNISTLikeQuantumIndexedDataset, torchvision.datasets.FashionMNIST)
