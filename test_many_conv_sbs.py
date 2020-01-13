import torch.nn as nn

from dctn.conv_sbs_spec import *
from dctn.conv_sbs import *


class TwoLayerModel(nn.Sequential):
    def __init__(self):
        super().__init__(
            ManyConvSBS(
                in_num_channels=3,
                in_quantum_dim_size=2,
                bond_dim_size=4,
                trace_edge=False,
                cores_specs=(
                    (
                        SBSSpecCore(Pos2D(-1, -1), 1),
                        # SBSSpecCore(Pos2D(-1, 0), 1),
                        # SBSSpecCore(Pos2D(-1, 1), 1),
                        # SBSSpecCore(Pos2D(0, 1), 2),
                        SBSSpecCore(Pos2D(0, 0), 2),
                        # SBSSpecCore(Pos2D(0, -1), 2),
                        # SBSSpecCore(Pos2D(1, -1), 1),
                        # SBSSpecCore(Pos2D(1, 0), 1),
                        SBSSpecCore(Pos2D(1, 1), 1),
                    ),
                    (
                        SBSSpecCore(Pos2D(-1, 1), 1),
                        SBSSpecCore(Pos2D(0, 0), 2),
                        SBSSpecCore(Pos2D(1, -1), 1),
                    ),
                    (
                        SBSSpecCore(Pos2D(-1, -1), 1),
                        SBSSpecCore(Pos2D(-1, 1), 1),
                        SBSSpecCore(Pos2D(1, 1), 2),
                        SBSSpecCore(Pos2D(1, -1), 1),
                    ),
                ),
            ),
            ConvSBS(
                SBSSpecString(
                    (
                        SBSSpecCore(Pos2D(0, 0), 1),
                        SBSSpecCore(Pos2D(0, 1), 2),
                        SBSSpecCore(Pos2D(1, 1), 1),
                        SBSSpecCore(Pos2D(1, 0), 1),
                    ),
                    (3, 3, 3, 1),
                    in_num_channels=3,
                    in_quantum_dim_size=2,
                )
            ),
        )


device = torch.device("cuda")
model = TwoLayerModel().to(device)
images = tuple(
    torch.rand(
        3,
        8,
        128,
        128,
        2,
        # names=("channel", "batch", "height", "width", "quantum"),
        device=device,
    )
)

from torch.optim import SGD
from torch.autograd import detect_anomaly
from torch import onnx

result = model(images)
# each tensor in result has shape batch × quantum × height × width
# oops, it should be batch height width quantum I guess
