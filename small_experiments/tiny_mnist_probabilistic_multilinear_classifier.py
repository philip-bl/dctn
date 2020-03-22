import random
import math

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

from libcrap.torch import set_random_seeds

MNIST_TRANSFORM = Compose((CenterCrop((16, 16)), Resize(4, 4), ToTensor()))

train_size = 50000
batch_size = 512
device = torch.device("cuda:1")
lr = 1e-2
num_iters = 30000
mov_avg_coeff = 0.99
seed = 0
save_where = (
    "/mnt/important/experiments/tiny_mnist_probabilistic_multilinear_classifier_adam.pth"
)

set_random_seeds(device, seed)
print(f"{seed=}")

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
X = rearrange(X, "b () h w -> (h w) b").to(device).double()
y = y.to(device)
X = torch.stack((torch.sin(X) ** 2, torch.cos(X) ** 2), dim=2)  # shape: 16×60000×2
assert torch.allclose(X.mean(), torch.tensor(0.5, device=device, dtype=torch.float64))
X_train = X[:, :train_size]
y_train = y[:train_size]
X_val = X[:, train_size:]
y_val = y[train_size:]

log_P = torch.log(
    torch.clamp(
        torch.randn((10,) + (2,) * 16, device=device) * 0.01 + 0.1,
        min=1e-8,
        max=1.0 - 1e-8,
    )
).double()
log_P.requires_grad_()


def forward(X_batch):
    result = oe.contract(
        "aijklmnopqrstuvwx,bi,bj,bk,bl,bm,bn,bo,bp,bq,br,bs,bt,bu,bv,bw,bx->ba",
        log_P.exp(),
        *(X_batch[i] for i in range(16)),
        optimize="auto",
    )
    assert result.dtype == torch.float64
    return result


@torch.no_grad()
def score(X, y):
    start = 0
    ce_loss = 0
    correct = 0
    while start < X.shape[1]:
        idx = slice(start, min(start + batch_size, X.shape[1]))
        X_batch = X[:, idx]
        y_batch = y[idx]
        unnormalized_probabilities = forward(X_batch)
        probabilities = unnormalized_probabilities / unnormalized_probabilities.sum(dim=1, keepdim=True)
        log_probabilities = torch.log(probabilities)        
        ce_loss += F.nll_loss(log_probabilities, y_batch, reduction="sum").item()
        correct += (probabilities.argmax(dim=1) == y_batch).sum().item()
        start += batch_size
    return ce_loss / X.shape[1], correct / X.shape[1]


ones = torch.ones(train_size, device=device)

optimizer = Adam((log_P,), lr)
for i in range(num_iters):
    batch_indices = torch.multinomial(ones, batch_size)
    X_batch = X[:, batch_indices]
    y_batch = y[batch_indices]
    unnormalized_probabilities = forward(X_batch)
    probabilities = unnormalized_probabilities / unnormalized_probabilities.sum(dim=1, keepdim=True)
    log_probabilities = torch.log(probabilities)
    train_ce_loss = F.nll_loss(log_probabilities, y_batch)
    train_ce_loss_avg = (
        train_ce_loss.item()
        if i == 0
        else train_ce_loss_avg * mov_avg_coeff
        + train_ce_loss.item() * (1 - mov_avg_coeff)
    )
    train_acc = (probabilities.argmax(dim=1) == y_batch).sum().float().item() / y_batch.shape[
        0
    ]
    train_acc_avg = (
        train_acc
        if i == 0
        else train_acc_avg * mov_avg_coeff + train_acc * (1 - mov_avg_coeff)
    )
    train_likelihood = torch.exp(-train_ce_loss).item()
    train_likelihood_avg = math.exp(-train_ce_loss_avg)
    print(
        f"{i=}: train_ce_loss={train_ce_loss.item():.4f}({train_ce_loss_avg:.4f}), {train_acc=:.4f}({train_acc_avg:.4f}), {train_likelihood=:.4f}({train_likelihood_avg:.4f})"
    )
    if i % 40 == 0:
        val_ce_loss, val_acc = score(X_val, y_val)
        val_likelihood = math.exp(-val_ce_loss)
        print(f"       {val_ce_loss=:.4f}, {val_acc=:.4f}, {val_likelihood=:.4f}")
    optimizer.zero_grad()
    train_ce_loss.backward()
    optimizer.step()
torch.save(log_P, save_where)
