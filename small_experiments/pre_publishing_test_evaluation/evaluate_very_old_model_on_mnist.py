from os import environ

environ["CUDA_VISIBLE_DEVICES"] = "1"

from math import pi
from pprint import pprint

import torch

from dctn.eps_plus_linear import EPSesPlusLinear, UnitTheoreticalOutputStd
from dctn.dataset_loading import get_mnist_data_loaders
from dctn.evaluation import score

# this path is for the model trained on 50k samples, it has Q_1=4
# MODEL_PATH = "/mnt/important/experiments/eps_plus_linear_mnist/2020-03-26T23-18-59/lightning_logs/version_0/checkpoints/epoch=526.ckpt"

# this path is for the model trained on very few samples, it has Q_1=32
MODEL_PATH = "/mnt/important/experiments/eps_plus_linear_mnist/2020-03-26T23-21-45/lightning_logs/version_0/checkpoints/epoch=40947.ckpt"


loaded = torch.load(MODEL_PATH, map_location="cpu")

pprint(loaded["hparams"])

sd = loaded["state_dict"]

print(sd.keys())

model = EPSesPlusLinear(
    ((4, 32),), UnitTheoreticalOutputStd(), 1.0, torch.device("cpu"), torch.float32
)

assert sd["eps.core"].shape == model.epses[0].shape
assert sd["linear.weight"].shape == model.linear.weight.shape
assert sd["linear.bias"].shape == model.linear.bias.shape

model.epses[0].data.copy_(sd["eps.core"])
model.linear.weight.data.copy_(sd["linear.weight"])
model.linear.bias.data.copy_(sd["linear.bias"])

model.eval()
model.to("cpu")

train_dl, val_dl, test_dl = get_mnist_data_loaders(
    "/mnt/hdd_1tb/datasets/mnist",
    128,
    torch.device("cpu"),
    (lambda X: 0.5 * (X * pi / 2.0).sin() ** 2, lambda X: 0.5 * (X * pi / 2.0).cos() ** 2),
)

print("train:", score(model, train_dl, torch.device("cpu")))
print("val:", score(model, val_dl, torch.device("cpu")))
print("test:", score(model, test_dl, torch.device("cpu")))
