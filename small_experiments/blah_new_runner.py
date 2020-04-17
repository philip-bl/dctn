from new_runner import main

main.main(
    (
        "--experiments-dir",
        "/tmp/new_runner_test/",
        "--ds-type",
        "fashionmnist",
        "--ds-path",
        "/mnt/hdd_1tb/datasets/fashionmnist",
        "--kernel-size",
        "4",
        "--out-size",
        "6",
        "--batch-size",
        "128",
        "--optimizer",
        "adam",
        "--lr",
        "1e-3",
    ),
    standalone_mode=False,
)
