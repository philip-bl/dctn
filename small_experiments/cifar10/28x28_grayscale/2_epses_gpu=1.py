from os import environ

environ["CUDA_VISIBLE_DEVICES"] = "1"

from new_runner import main

main.main(
    (
        "--ds-type",
        "cifar10_28x28_grayscale",
        "--ds-path",
        "/mnt/hdd_1tb/datasets/cifar10",
        "--batch-size",
        "128",
        "--optimizer",
        "adam",
        "--experiments-dir",
        "/mnt/important/experiments/cifar10/28x28_grayscale/2_epses",
        "--epses-specs",
        "(4,4),(3,6)",
        "--lr",
        "1.5e-4",
        "--reg-type",
        "epses_composition",
        "--reg-coeff",
        "0.",
        "--phi-multiplier",
        "1.24",
        "--init-eps-zero-centered-normal-std",
        "0",
        "3.9069653e-3",
        "--init-eps-zero-centered-normal-std",
        "1",
        "1.9525534e-3",
        "--init-linear-weight-zero-centered-uniform",
        "0.01774992567234187",
        "--init-linear-bias-zero-centered-uniform",
        "0.01774992567234187",
    ),
    standalone_mode=False,
)
