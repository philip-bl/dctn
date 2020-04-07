from typing import *

import torch
import torch.nn.functional as F

from dctn.eps_plus_linear import EPSPlusLinear
from dctn.dataset_loading import get_fashionmnist_data_loaders
from dctn.evaluation import score

# good model:
# load_path = "/mnt/important/experiments/eps_plus_linear_fashionmnist/2020-03-27T16-42-28/lightning_logs/version_0/checkpoints/epoch=1114.ckpt"
ds_path = "/mnt/hdd_1tb/datasets/fashionmnist" # ds stands for dataset
device = torch.device("cuda")
bs = 128 # batch size

def load_old_eps_plus_linear(path: str) -> EPSPlusLinear:
  """Here I load the model in the old format. There are 2 caveats:
  1. Its keys have different names
  2. During inputs preprocessing, instead of multiplying sin^2 and cos^2 by 2,
     I messed up and divided them by 2, which gives a scaling factor of 4."""
  keys_mapping = {"eps.core": "0.core", "linear.weight": "2.weight", "linear.bias": "2.bias"}
  state_dict = {keys_mapping[k]: v for k, v in torch.load(load_path)["state_dict"].items()}
  kernel_size = 4
  out_size = 4
  model = EPSPlusLinear(kernel_size, out_size)
  model.load_state_dict(state_dict)
  model[0].core.data /= 4 ** (kernel_size**2)
  return model

model = load_old_eps_plus_linear(load_path).to(device)
train_dl, val_dl, test_dl = get_fashionmnist_data_loaders(ds_path, bs, device)

print("On train loss={0:.4f}, accuracy={1:.2%}".format(*score(model, train_dl, device))) # 0.1896, 93.39%
print("On val   loss={0:.4f}, accuracy={1:.2%}".format(*score(model, val_dl, device)))   # 0.2752, 90.01%
print("On test  loss={0:.4f}, accuracy={1:.2%}".format(*score(model, test_dl, device)))  # 0.2996, 89.72%
