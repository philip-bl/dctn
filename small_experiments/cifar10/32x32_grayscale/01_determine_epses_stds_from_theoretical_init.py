from os import environ

environ["CUDA_VISIBLE_DEVICES"] = "0"

from new_runner import main

main.main(
    (
        "--ds-type",
        "cifar10_32x32_grayscale",
        "--ds-path",
        "/mnt/hdd_1tb/datasets/cifar10",
        "--batch-size",
        "128",
        "--optimizer",
        "adam",
        "--experiments-dir",
        "/mnt/important/experiments/cifar10/32x32_grayscale/2_epses",
        "--epses-specs",
        "(4,4),(3,6)",
        "--lr",
        "1.5e-4",
        "--reg-type",
        "epses_composition",
        "--reg-coeff",
        "0.",
        "--init-epses-composition-unit-theoretical-output-std",
    ),
    standalone_mode=False,
)
