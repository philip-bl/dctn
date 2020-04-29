from random import shuffle, seed
from os import environ
from os.path import expanduser
from subprocess import Popen
from time import sleep
from typing import Tuple, Any

import numpy as np

min_lr = 2e-5
max_lr = 5e-3
num_experiments = 16

seed(0)
lrs = list(np.logspace(np.log10(min_lr), np.log10(max_lr), num=num_experiments))
shuffle(lrs)
assert len(lrs) >= 2

common_args = (
    "python",
    expanduser("~/projects/dctn/new_runner.py"),
    "--experiments-dir",
    "/mnt/important/experiments/2_epses_plus_linear_fashionmnist/making_empirical_std_one_lr_finding/",
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
    "--patience",
    "10",
    "--reg-type",
    "epses_composition",
    "--reg-coeff",
    "1e-2",
    "--no-es-train-acc",
    "--no-es-train-mean-ce",
    "--no-breakpoint-on-nan-loss",
    "--init-epses-composition-unit-empirical-output-std",
)


def make_args(lr: float) -> Tuple[str, ...]:
    return (*common_args, "--lr", str(lr))


def create_process(device: int):
    lr = lrs.pop()
    print(f"{lr=} popped with {device=}")
    p = Popen(make_args(lr), env={**environ, "CUDA_VISIBLE_DEVICES": str(device)})
    sleep(1.5)  # otherwise I get filename clashes
    return p


ps = [create_process(i) for i in range(2)]

while True:
    for i in range(2):
        if (retcode := ps[i].poll()) is not None:
            if retcode != 0:
                print("error!\n" * 50)
            if len(lrs) != 0:
                ps[i] = create_process(i)
    if len(lrs) == 0 and all(p.poll() is not None for p in ps):
        break
    sleep(10)
