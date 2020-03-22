import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop
from torch.optim import Adam


from einops import rearrange

import opt_einsum as oe

MNIST_TRANSFORM = Compose((CenterCrop((16, 16)), Resize(4, 4), ToTensor()))

train_size = 50000
batch_size = 512
device = torch.device("cuda")
lr = 5e-3
num_iters = 100000
mov_avg_coeff = 0.99

X, y = next(
    iter(
        DataLoader(
            MNIST(
                "/mnt/hdd_1tb/datasets/mnist/dataset_source",
                train=True,
                download=False,
                transform=MNIST_TRANSFORM,
            ),
            shuffle=True,
            batch_size=60000,
        )
    )
)
X = rearrange(X, "b () h w -> (h w) b").to(device)
y = y.to(device)
X = torch.stack((torch.sin(X) ** 2, torch.cos(X) ** 2), dim=2)  # shape: 16×60000×2
X_train = X[:, :train_size]
y_train = y[:train_size]
X_val = X[:, train_size:]
y_val = y[train_size:]

W = torch.randn((10,) + (2,) * 16, device=device)
W *= 0.5
W.requires_grad_()


def forward(X_batch):
    return oe.contract(
        "aijklmnopqrstuvwx,bi,bj,bk,bl,bm,bn,bo,bp,bq,br,bs,bt,bu,bv,bw,bx->ba",
        W,
        *(X_batch[i] for i in range(16)),
        optimize="auto",
    )


@torch.no_grad()
def score(X, y):
    start = 0
    ce_loss = 0
    correct = 0
    while start < X.shape[1]:
        idx = slice(start, min(start + batch_size, X.shape[1]))
        X_batch = X[:, idx]
        y_batch = y[idx]
        logits = forward(X_batch)
        ce_loss += F.cross_entropy(logits, y_batch, reduction="sum").item()
        correct += (logits.argmax(dim=1) == y_batch).sum().item()
        start += batch_size
    return ce_loss / X.shape[1], correct / X.shape[1]

ones = torch.ones(train_size, device=device)

optimizer = Adam((W,), lr)
for i in range(num_iters):
    batch_indices = torch.multinomial(ones, batch_size)
    X_batch = X[:, batch_indices]
    y_batch = y[batch_indices]
    logits = forward(X_batch)
    train_ce_loss = F.cross_entropy(logits, y_batch)
    train_ce_loss_avg = (
        train_ce_loss.item()
        if i == 0
        else train_ce_loss_avg * mov_avg_coeff
        + train_ce_loss.item() * (1 - mov_avg_coeff)
    )
    train_acc = (logits.argmax(dim=1) == y_batch).sum().float().item() / y_batch.shape[
        0
    ]
    train_acc_avg = (
        train_acc
        if i == 0
        else train_acc_avg * mov_avg_coeff + train_acc * (1 - mov_avg_coeff)
    )
    print(
        f"{i=}: train_ce_loss={train_ce_loss.item():.4f}({train_ce_loss_avg:.4f}), {train_acc=:.4f}({train_acc_avg:.4f})"
    )
    if i % 40 == 0:
        val_ce_loss, val_acc = score(X_val, y_val)
        print(f"       {val_ce_loss=}, {val_acc=}")
    optimizer.zero_grad()
    # if W.grad is not None:
    #     W.grad.zero_()
    train_ce_loss.backward()
    optimizer.step()
    # W.data -= lr * W.grad
