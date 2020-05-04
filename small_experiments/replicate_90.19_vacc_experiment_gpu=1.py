from os import environ

environ["CUDA_VISIBLE_DEVICES"] = "1"

from new_runner import main

main.main(
    (
        # seed is specific to this file only
        "--seed",
        "1",
        # settings which are basically the same for all experiments
        "--ds-type",
        "fashionmnist",
        "--ds-path",
        "/mnt/hdd_1tb/datasets/fashionmnist",
        "--batch-size",
        "128",
        "--optimizer",
        "adam",
        # settings introduced earlier which are unique to this experiment
        "--experiments-dir",
        "/mnt/important/experiments/eps_plus_linear_fashionmnist/replicate_90.19_vacc",
        "--epses-specs",
        "(4,4)",
        "--lr",
        "3e-3",
        "--reg-type",
        "epses_composition",
        "--reg-coeff",
        "0.",
        # super duper new settings which might bug out
        "--phi-multiplier",
        "0.5",
        "--init-epses-zero-centered-normal",
        "0.25",
        "--init-linear-weight-zero-centered-uniform",
        "0.02",
        "--init-linear-bias-zero-centered-uniform",
        "0.02",
    ),
    standalone_mode=False,
)
