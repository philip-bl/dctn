from dctn.conv_sbs_spec import *
from dctn.conv_sbs import *

cores_specs = (
    SBSSpecCore(Pos2D(-1, -1), 1),
    # SBSSpecCore(Pos2D(-1, 0), 1),
    # SBSSpecCore(Pos2D(-1, 1), 1),
    # SBSSpecCore(Pos2D(0, 1), 2),
    SBSSpecCore(Pos2D(0, 0), 2),
    # SBSSpecCore(Pos2D(0, -1), 2),
    # SBSSpecCore(Pos2D(1, -1), 1),
    # SBSSpecCore(Pos2D(1, 0), 1),
    # SBSSpecCore(Pos2D(1, 1), 1),
)

string_spec = SBSSpecString(
    cores_specs,
    (1,) + (4,) * (len(cores_specs) - 1),
    in_num_channels=3,
    in_quantum_dim_size=2,
)

string = ConvSBS(string_spec)

device = torch.device("cpu")
string.to(device)
channels = tuple(torch.rand(
    3,
    8,
    128,
    128,
    2,
    # names=("channel", "batch", "height", "width", "quantum"),
    device=device,
))

from torch.optim import SGD
from torch.autograd import detect_anomaly
from torch import onnx

#onnx.export(string, (channels,), "two_cores_string.onnx", verbose=True)
# to fix
# RuntimeError: Unsupported prim::Constant kind: `s`. Send a bug report.
# I can use the master branch of pytorch. It seems there are problems with onnx adding padding of const size

optimizer = SGD(string.parameters(), lr=3.0)

with detect_anomaly():
    for i in range(50):
        y = string(channels)
        loss = torch.mean((y - 0.1)**2)
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
