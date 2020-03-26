import random

import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop

from einops import rearrange

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

MNIST_TRANSFORM = Compose((
  CenterCrop((16, 16)),
  Resize(4, 4),
  ToTensor()
))



X, y = next(iter(DataLoader(MNIST(
  "/mnt/hdd_1tb/datasets/mnist/dataset_source",
  train=True,
  download=False,
  transform=MNIST_TRANSFORM,
), shuffle=True, batch_size=60000)))
X = rearrange(X, "b () h w -> b (h w)").numpy()
train_size = 50000
X_train = X[:train_size]
y_train = y[:train_size]
X_val = X[train_size:]
y_val = y[train_size:]
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
clf = LogisticRegression(max_iter=100000).fit(X_train, y_train)
train_acc = clf.score(X_train, y_train)
val_acc = clf.score(X_val, y_val)
print(f"{train_acc=}, {val_acc=}")
