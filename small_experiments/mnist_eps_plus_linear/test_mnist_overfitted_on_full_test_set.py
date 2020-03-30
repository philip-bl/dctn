import torch
import torch.nn as nn

class EPSPlusLinear(nn.Module):
  


device = torch.device("cuda")
sd = torch.load("/mnt/important/experiments/eps_plus_linear_mnist/good_model/epspluslinear_state_dict.pth", map_location=device)
