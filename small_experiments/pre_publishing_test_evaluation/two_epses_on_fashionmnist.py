from math import pi
from pprint import pprint

import torch

from dctn.eps_plus_linear import EPSesPlusLinear, UnitTheoreticalOutputStd
from dctn.dataset_loading import get_fashionmnist_data_loaders
from dctn.evaluation import score


MODEL_PATH = "/mnt/important/experiments/2_epses_plus_linear_fashionmnist/adam_and_epswise_l2/2020-04-26T23:06:19_earlystopped/model_best_val_acc_nitd=0002000_tracc=0.9697_vacc=0.8820_trmce=0.0887_vmce=0.4714.pth"

device = torch.device("cuda:0")

model = EPSesPlusLinear(
    ((4, 4), (3, 6)), UnitTheoreticalOutputStd(), 1.0, device, torch.float32
)

sd = torch.load(MODEL_PATH, map_location=device)
remapped_sd = {
    {
        "0.core": "epses.0",
        "2.core": "epses.1",
        "4.weight": "linear.weight",
        "4.bias": "linear.bias",
    }[key]: value
    for key, value in sd.items()
}

model.load_state_dict(remapped_sd, strict=False)

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
print("val:", score(model, val_dl, device))  # 88.2%
print("test:", score(model, test_dl, device))  # 87.65%
