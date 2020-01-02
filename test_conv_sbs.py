from dctn.conv_sbs_spec import *
from dctn.conv_sbs import *

string_spec = SBSSpecString(
    (
        SBSSpecCore(Pos2D(-1, -1), 1),
        SBSSpecCore(Pos2D(-1, 1), 2),
        SBSSpecCore(Pos2D(1, -1), 2),
        SBSSpecCore(Pos2D(1, 1), 1),
    ),
    (1, 4, 4, 4),
    in_num_channels=3,
    in_quantum_dim_size=2,
)

string = ConvSBS(string_spec)

string.gen_einsum_exprs(8, 112, 112)

for expr in string._first_stage_einsum_exprs:
    print(expr)


device = torch.device("cuda")
string.to(device)
images = torch.rand(
    8,
    3,
    128,
    128,
    2,
    names=("batch", "channel", "height", "width", "quantum"),
    device=device,
)
y = string(images)
