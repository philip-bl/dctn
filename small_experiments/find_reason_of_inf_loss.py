from new_runner import main

main.main(
    (
        "--device",
        "cuda",
        "--experiments-dir",
        "/mnt/important/experiments/2_epses_plus_linear_fashionmnist/figuring_out_inf_with_sgd",
        "--ds-type",
        "fashionmnist",
        "--ds-path",
        "/mnt/hdd_1tb/datasets/fashionmnist",
        "--epses-specs",
        "(4,4),(3,6)",
        "--batch-size",
        "128",
        "--no-es-train-acc",
        "--no-es-train-mean-ce",
        "--optimizer",
        "sgd",
        "--lr",
        "1.5000000000000004e-05",
        "--tb-batches",
        "--no-breakpoint-on-nan-loss",
    ),
    standalone_mode=False,
)
