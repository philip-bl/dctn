from math import pi
from pprint import pprint

import torch

from dctn.eps_plus_linear import EPSesPlusLinear, UnitTheoreticalOutputStd
from dctn.dataset_loading import get_fashionmnist_data_loaders
from dctn.evaluation import score


MODEL_PATH = "/mnt/important/experiments/3_epses_plus_linear_fashionmnist/2020-05-12T19:33:11_vacc=0.7708_manually_stopped/model_best_val_acc_nitd=0430000_tracc=0.8088_vacc=0.7708_trmce=0.8494_vmce=145.3116.pth"

device = torch.device("cuda:0")

model = EPSesPlusLinear(
    ((4, 4), (3, 12), (2, 24)), UnitTheoreticalOutputStd(), 1.0, device, torch.float32
)

sd = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(sd)

model.eval()
model.to(device)

train_dl, val_dl, test_dl = get_fashionmnist_data_loaders(
    "/mnt/hdd_1tb/datasets/fashionmnist",
    32,
    device,
    (
        lambda X: 1.45646 * (X * pi / 2.0).sin() ** 2,
        lambda X: 1.45646 * (X * pi / 2.0).cos() ** 2,
    ),
)

# print("train:", score(model, train_dl, device))  # takes too long on CPU
print("val:", score(model, val_dl, device))  # 77.08%
print("test:", score(model, test_dl, device))  # 75.94%
