from os import environ
environ["CUDA_VISIBLE_DEVICES"] = "1"

from new_runner import main

main.main(
  (
    "--experiments-dir", "/mnt/important/experiments/eps_plus_linear_fashionmnist",
    "--ds-type", "fashionmnist",
    "--ds-path", "/mnt/hdd_1tb/datasets/fashionmnist",
    "--kernel-size", "4",
    "--out-size", "4",
    "--batch-size", "128",
    "--optimizer", "adam",
    "--lr", "0.003",
    "--old-scaling"
  ),
  standalone_mode=False)
