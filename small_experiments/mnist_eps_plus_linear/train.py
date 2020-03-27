import math
import statistics
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from einops import rearrange
from dctn.eps import EPS

class EPSPlusLinear(pl.LightningModule):
  def __init__(self, hparams):
    super().__init__()
    self.hparams = hparams
    self.eps = EPS(hparams.kernel_size, 1, 2, hparams.out_size)
    self.linear = nn.Linear(hparams.out_size * (28-hparams.kernel_size+1)**2, 10, bias=True)

  def forward(self, batch: torch.Tensor) -> torch.Tensor:
    batch = batch.squeeze(1)
    assert batch.ndim == 3
    batch *= math.pi / 2
    sin_squared = torch.sin(batch) ** 2 / 2 # shape: batch×height×width
    cos_squared = torch.cos(batch) ** 2 / 2
    concatenated = torch.cat((sin_squared.unsqueeze(3), cos_squared.unsqueeze(3)), dim=3) \
      .unsqueeze(0) # 1×batch×height×width×2
    intermediate = self.eps(concatenated)
    return self.linear(rearrange(intermediate, "b h w q -> b (h w q)"))

  def train_dataloader(self):
    return DataLoader(
      Subset(MNIST(self.hparams.dataset_root, transform=ToTensor()), range(50000)),
      batch_size=self.hparams.batch_size, shuffle=True, drop_last=True)

  def val_dataloader(self):
    return DataLoader(
      Subset(MNIST(self.hparams.dataset_root, transform=ToTensor()), range(50000, 60000)),
      batch_size=self.hparams.batch_size)

  def test_dataloader(self):
    return DataLoader(
      MNIST(self.hparams.dataset_root, train=False, transform=ToTensor()),
      batch_size=self.hparams.batch_size)

  def configure_optimizers(self):
    return Adam(self.parameters(), lr=self.hparams.learning_rate)

  def training_step(self, batch, batch_idx):
    x, y = batch
    logits = self(x)
    loss = F.cross_entropy(logits, y)
    with torch.no_grad():
      accuracy = (torch.argmax(logits, dim=1) == y).float().mean()
    return {"loss": loss, "accuracy": accuracy,
      "log": {"train_loss": loss, "train_accuracy": accuracy}}

  def validation_step(self, batch, batch_idx):
    x, y = batch
    logits = self(x)
    loss = F.cross_entropy(logits, y)
    num_correct = (torch.argmax(logits, dim=1) == y).float().sum()
    return {"loss": loss, "num_correct": num_correct, "batch_size": len(y)}

  def validation_epoch_end(self, outputs):
    non_reduced_losses = (record["loss"].item() * record["batch_size"] for record in outputs)
    num_samples = sum(record["batch_size"] for record in outputs)
    loss = sum(non_reduced_losses) / num_samples
    num_correct = sum(record["num_correct"].item() for record in outputs)
    accuracy = num_correct / num_samples
    return {"loss": loss, "accuracy": accuracy,
      "log": {"val_loss": loss, "val_accuracy": accuracy},
      "progress_bar": {"val_loss": loss, "val_accuracy": accuracy}}

  def test_step(self, batch, batch_idx):
    return self.validation_step(batch, batch_idx)

  def test_epoch_end(self, outputs):
    outputs = self.validation_epoch_end(outputs)
    return {"loss": outputs["loss"], "accuracy": outputs["accuracy"],
      "progress_bar": {"test_loss": outputs["loss"], "test_accuracy": outputs["accuracy"]}}

  @staticmethod
  def add_model_specific_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser])
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--kernel_size", type=int)
    parser.add_argument("--out_size", type=int)
    parser.add_argument("--dataset_root", type=str)
    return parser


def main(hparams):
  model = EPSPlusLinear(hparams)
  trainer = pl.Trainer(max_epochs=100000, gpus=1, default_save_path=hparams.save_path,
    overfit_pct=hparams.overfit_pct, weights_summary="full", print_nan_grads=True, track_grad_norm=2)
  trainer.fit(model)


if __name__ == "__main__":
  parser = ArgumentParser(add_help=False)
  parser.add_argument("--save_path", type=str)
  parser.add_argument("--overfit_pct", type=float, default=0.0)
  parser = EPSPlusLinear.add_model_specific_args(parser)
  hparams = parser.parse_args()
  main(hparams)
