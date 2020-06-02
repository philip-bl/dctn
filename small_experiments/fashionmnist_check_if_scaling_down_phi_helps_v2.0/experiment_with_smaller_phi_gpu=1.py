from os import environ

environ["CUDA_VISIBLE_DEVICES"] = "1"

from new_runner import main

main.main(
    (
        "--ds-type",
        "fashionmnist",
        "--ds-path",
        "/mnt/hdd_1tb/datasets/fashionmnist",
        "--batch-size",
        "128",
        "--optimizer",
        "adam",
        "--experiments-dir",
        "/mnt/important/experiments/2_epses_plus_linear_fashionmnist/2020-05-18_fashionmnist_check_if_scaling_down_phi_helps_v2.0",
        "--epses-specs",
        "(4,4),(3,6)",
        "--lr",
        "1.5e-4",
        "--reg-type",
        "epses_composition",
        "--reg-coeff",
        "1e-2",
        "--phi-multiplier",
        "1.35",
        "--init-eps-zero-centered-normal-std",
        "0",
        "5.885948427021503448486328125000e-03",
        "--init-eps-zero-centered-normal-std",
        "1",
        "2.112586298608221113681793212891e-05",
        "--init-linear-weight-zero-centered-normal-std",
        "4.437481418085467352319106737468e-03",
        "--init-linear-bias-zero-centered-uniform",
        "1.77499256723418694092764269498729845508933067321777343750e-02",
    ),
    standalone_mode=False,
)
