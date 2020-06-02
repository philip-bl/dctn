from os import environ

environ["CUDA_VISIBLE_DEVICES"] = "0"

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
        "--init-epses-composition-unit-empirical-output-std",
    ),
    standalone_mode=False,
)
