from dctn.conv_sbs_spec import Pos2D, SBSSpecCore, SBSSpecString

def test_SBSSpecString_all_dangling_dim_names() -> None:
  spec = SBSSpecString(
    (SBSSpecCore(Pos2D(0, 0), 1), SBSSpecCore(Pos2D(0, 1), 1),
     SBSSpecCore(Pos2D(1, 1), 2), SBSSpecCore(Pos2D(1, 0), 1)),
    bond_sizes=(5, 5, 5, 5), in_num_channels=3, in_quantum_dim_size=100)
  assert spec.all_dangling_dim_names == (
    # rows are indices of cores, columns are indices of input channels
    "in_quantum_0_0", "in_quantum_1_0", "in_quantum_2_0",
    "in_quantum_0_1", "in_quantum_1_1", "in_quantum_2_1",
    "in_quantum_0_2", "in_quantum_1_2", "in_quantum_2_2",
    "in_quantum_0_3", "in_quantum_1_3", "in_quantum_2_3",

    "out_quantum_0",
    "out_quantum_1",
    "out_quantum_2",
    "out_quantum_3")
