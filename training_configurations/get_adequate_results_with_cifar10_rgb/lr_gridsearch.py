from random import shuffle, seed, randint
import itertools
from os import environ
from os.path import expanduser
from subprocess import Popen
from time import sleep
from typing import Tuple, Any

import numpy as np

num_points = 10
lrs = list(str(x) for x in np.logspace(-6, -2.5, num_points))
reg_coeffs = list(str(x) for x in np.logspace(-12, 0, num_points))

configs = [
    {"--lr": lr, "--reg-coeff": reg_coeff}
    for lr, reg_coeff in itertools.product(lrs, reg_coeffs)
]


shuffle(configs)

common_args = (
    "python",
    expanduser("~/projects/dctn/new_runner.py"),
    "--experiments-dir",
    "/mnt/important/experiments/cifar10/ycbcr_plus_constant_channel",
    "--ds-type",
    "cifar10_ycbcr",
    "--ds-path",
    "/mnt/hdd_1tb/datasets/cifar10",
    "--epses-specs",
    "(3,6)",
    "--batch-size",
    "128",
    "--optimizer",
    "adam",
    "--eval-schedule",
    "((10,2), (20,3), (40,8), (200,20), (500,100), (2000,200), (6000,500), (None,2000))",
    "--patience",
    "60",
    "--reg-type",
    "epses_composition",
    "--no-es-train-acc",
    "--no-es-train-mean-ce",
    "--no-breakpoint-on-nan-loss",
    "--init-epses-composition-unit-empirical-output-std",
    "--max-num-iters",
    "300000",
    "--center-and-normalize-each-channel",
    "--add-constant-channel",
    "1.",
)


def make_args(config) -> Tuple[str, ...]:
    return (
        *common_args,
        "--seed",
        str(randint(0, 65535)),
        *itertools.chain.from_iterable(config.items()),
    )


def create_process(device: int):
    config = configs.pop()
    print(f"{config=} popped with {device=}")
    process = Popen(make_args(config), env={**environ, "CUDA_VISIBLE_DEVICES": str(device)})
    sleep(1.5)  # otherwise I get filename clashes
    return process


processes = [create_process(i) for i in range(2)]

while True:
    for i in range(2):
        if (retcode := processes[i].poll()) is not None:
            if retcode != 0:
                print("error!\n" * 50)
            if len(processes) != 0:
                processes[i] = create_process(i)
    if len(configs) == 0 and all(process.poll() is not None for process in processes):
        break
    sleep(10)
