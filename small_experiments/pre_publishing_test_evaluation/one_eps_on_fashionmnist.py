from math import pi
from pprint import pprint

import torch

from dctn.eps_plus_linear import EPSesPlusLinear, UnitTheoreticalOutputStd
from dctn.dataset_loading import get_fashionmnist_data_loaders
from dctn.evaluation import score

MODEL_PATH = "/mnt/important/experiments/eps_plus_linear_fashionmnist/replicate_90.19_vacc/2020-05-04T23:13:52_stopped_manually/model_best_val_acc_nitd=0580000_tracc=0.9456_vacc=0.9025_trmce=0.1624_vmce=0.2738.pth"

device = torch.device("cpu")

model = EPSesPlusLinear(((4, 4),), UnitTheoreticalOutputStd(), 1.0, device, torch.float32)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

model.eval()
model.to(device)

train_dl, val_dl, test_dl = get_fashionmnist_data_loaders(
    "/mnt/hdd_1tb/datasets/fashionmnist",
    128,
    device,
    (lambda X: 0.5 * (X * pi / 2.0).sin() ** 2, lambda X: 0.5 * (X * pi / 2.0).cos() ** 2),
)

print("train:", score(model, train_dl, device))
print("val:", score(model, val_dl, device))
print("test:", score(model, test_dl, device))
