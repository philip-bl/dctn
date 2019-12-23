from dctn.conv_sbs_spec import *
from dctn.conv_sbs import *

string_spec = SBSSpecString(
    (SBSSpecCore((-1, 1), 1), SBSSpecCore((0, 0), 2), SBSSpecCore((1, -1), 1)),
    (1, 4, 4),
    in_num_channels=10,
    in_quantum_dim_size=2,
)

string = ConvSBS(string_spec)

string.gen_einsum_exprs(8, 112, 112)

for expr in string._first_stage_einsum_exprs:
    print(expr)
