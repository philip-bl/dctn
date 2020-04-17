import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.transforms import ToTensor, Normalize, Compose

from ignite.engine import (
    Engine,
    create_supervised_trainer,
    create_supervised_evaluator,
    Events,
)
from ignite.contrib.handlers.tqdm_logger import ProgressBar
import ignite.metrics

from einops.layers.torch import Rearrange


def dataset_with_indices(cls):
    """
  Returns a modified class cls, which returns tuples like (X, y, indices) instead of just (X, y).
  """

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {"__getitem__": __getitem__})


FashionMNIST = dataset_with_indices(FashionMNIST)

model = nn.Sequential(Rearrange("b () h w -> b (h w)"), nn.Linear(28 ** 2, 10))
DATASET_PATH = "/mnt/hdd_1tb/datasets/fashionmnist"
dataset = FashionMNIST(
    DATASET_PATH, transform=Compose((ToTensor(), Normalize((0.286,), (0.353,))))
)
TRAIN_SIZE = 50000
BATCH_SIZE = 512
DEVICE = torch.device("cuda")
train_dataloader = DataLoader(
    dataset,
    BATCH_SIZE,
    sampler=SubsetRandomSampler(range(TRAIN_SIZE)),
    pin_memory=(DEVICE.type == "cuda"),
    drop_last=True,
)
val_dataloader = DataLoader(
    dataset,
    BATCH_SIZE,
    sampler=SubsetRandomSampler(range(TRAIN_SIZE, 60000)),
    pin_memory=(DEVICE.type == "cuda"),
)
optimizer = Adam(model.parameters(), lr=3e-4)


def prepare_batch(batch, device, non_blocking) -> Tuple[torch.Tensor, torch.Tensor]:
    X, y, indices = batch
    return X.to(device), y.to(device)


trainer = create_supervised_trainer(
    model, optimizer, F.cross_entropy, DEVICE, prepare_batch=prepare_batch
)
evaluator = create_supervised_evaluator(
    model, {"loss": ignite.metrics.Loss(F.cross_entropy)}, DEVICE, prepare_batch=prepare_batch
)


def foo(_):
    print(f"Before iteration {trainer.state.iteration} (counting from 1):")
    evaluator.run(train_dataloader, epoch_length=math.ceil(10000 / BATCH_SIZE))
    print(f"\tEstimate of train loss = {evaluator.state.metrics['loss']}")
    evaluator.run(val_dataloader, epoch_length=math.ceil(10000 / BATCH_SIZE))
    print(f"\tEstimate of  val   loss = {evaluator.state.metrics['loss']}")


trainer.on(Events.ITERATION_STARTED(once=True))(foo)
trainer.on(Events.ITERATION_STARTED(every=15))(foo)


@trainer.on(Events.EPOCH_COMPLETED)
def print_first_index(_):
    print(f"{trainer.state.batch[2][0]=}")


trainer.run(train_dataloader, 3)
