from os import environ

environ["CUDA_VISIBLE_DEVICES"] = "1"

from new_runner import main

# copies /mnt/important/experiments/2_epses_plus_linear_fashionmnist/2020-04-15T19:42:03/
# but with epswise l2 regularization
main.main(
    (
        "--experiments-dir",
        "/mnt/important/experiments/2_epses_plus_linear_fashionmnist/adam_and_comp_reg",
        "--ds-type",
        "fashionmnist",
        "--ds-path",
        "/mnt/hdd_1tb/datasets/fashionmnist",
        "--epses-specs",
        "(4,4),(3,6)",
        "--batch-size",
        "128",
        "--optimizer",
        "adam",
        "--lr",
        "1.11e-4",
        "--reg-type",
        "epses_composition",
        "--reg-coeff",
        "1e-3",
        "--no-es-train-acc",
        "--no-es-train-mean-ce",
    ),
    standalone_mode=False,
)
