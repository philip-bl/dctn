from os import environ

environ["CUDA_VISIBLE_DEVICES"] = "0"

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
        "/mnt/important/experiments/cifar10/28x28_grayscale/1_eps_k=4_q=4_very_small_stds",
        "--epses-specs",
        "(4,4)",
        "--lr",
        "3e-3",
        "--reg-type",
        "epses_composition",
        "--reg-coeff",
        "0.",
        "--init-eps-zero-centered-normal-std", "0", "0.25",
        "--init-linear-bias-zero-centered-uniform", "0.02",
        "--init-linear-weight-zero-centered-uniform": "0.02"
    ),
    standalone_mode=False,
)
