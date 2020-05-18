from random import shuffle, seed, randint
import itertools
from os import environ
from os.path import expanduser
from subprocess import Popen
from time import sleep
from typing import Tuple, Any

import numpy as np

num_experiments = 10
lrs = list(np.logspace(-7.5, -4.3, num_experiments))

shuffle(lrs)
assert len(lrs) >= 2

common_args = (
    "python",
    expanduser("~/projects/dctn/new_runner.py"),
    "--experiments-dir",
    "/mnt/important/experiments/cifar10/rgb/lr_gridsearch/",
    "--ds-type",
    "cifar10_rgb",
    "--ds-path",
    "/mnt/hdd_1tb/datasets/cifar10",
    "--epses-specs",
    "(3,4),(3,6)",
    "--batch-size",
    "128",
    "--optimizer",
    "adam",
    "--eval-schedule",
    "((10,1), (20,2), (40,4), (200,10), (500,50), (2000,100), (6000,300), (None,1000))",
    "--patience",
    "60",
    "--reg-type",
    "epses_composition",
    "--reg-coeff",
    "1e-1",
    "--no-es-train-acc",
    "--no-es-train-mean-ce",
    "--no-breakpoint-on-nan-loss",
    "--init-epses-composition-unit-empirical-output-std",
    "--max-num-iters",
    "200000",
)


def make_args(lr: float) -> Tuple[str, ...]:
    return (*common_args, "--lr", str(lr), "--seed", str(randint(0, 65535)))


def create_process(device: int):
    lr = lrs.pop()
    print(f"{lr=:.3e} popped with {device=}")
    process = Popen(make_args(lr), env={**environ, "CUDA_VISIBLE_DEVICES": str(device)})
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
    if len(lrs) == 0 and all(process.poll() is not None for process in processes):
        break
    sleep(10)
