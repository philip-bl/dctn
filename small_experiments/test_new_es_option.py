from new_runner import main

main.main(
    (
        "--experiments-dir",
        "/mnt/important/experiments/2_epses_plus_linear_fashionmnist",
        "--ds-type",
        "fashionmnist",
        "--ds-path",
        "/mnt/hdd_1tb/datasets/fashionmnist",
        "--epses-specs",
        "(3,4),(3,6)",
        "--batch-size",
        "128",
        "--optimizer",
        "adam",
        "--lr",
        "1.5e-3",
        "--es-metrics",
        "('val_acc', 'val_mean_ce')",
        "--patience",
        "3",
    ),
    standalone_mode=False,
)
