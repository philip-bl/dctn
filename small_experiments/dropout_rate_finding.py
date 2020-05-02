from random import shuffle, seed
import itertools
from os import environ
from os.path import expanduser
from subprocess import Popen
from time import sleep
from typing import Tuple, Any

import numpy as np

min_p = 0.01
max_p = 0.4
num_experiments = 5
additional_ps = [1.0]

ps = [0.05, 0.2, 0.4]
lrs = [1.821e-4, 9e-5, 4.5e-5]

additional_p_and_lr = (1.0, 1.821e-4)

ps_and_lrs = [*itertools.product(ps, lrs), additional_p_and_lr]

seed(0)
shuffle(ps_and_lrs)
assert len(ps_and_lrs) >= 2

common_args = (
    "python",
    expanduser("~/projects/dctn/new_runner.py"),
    "--seed",
    "1",  # NOTICE: NOT THE USUAL "0" SEED BECAUSE I SUSPECT RANDOM MEANS A LOT HERE
    "--experiments-dir",
    "/mnt/important/experiments/2_epses_plus_linear_fashionmnist/dropout_rate_finding",
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
    "--eval-schedule",
    "((10,1), (20,2), (40,4), (200,10), (500,50), (2000,100), (6000,300), (20000,1000), (None,2500))",
    "--patience",
    "30",
    "--reg-type",
    "epses_composition",
    "--reg-coeff",
    "1e-2",
    "--no-es-train-acc",
    "--no-es-train-mean-ce",
    "--no-breakpoint-on-nan-loss",
    "--init-epses-composition-unit-empirical-output-std",
)


def make_args(p: float, lr: float) -> Tuple[str, ...]:
    return (*common_args, "--dropout-p", str(p), "--lr", str(lr))


def create_process(device: int):
    p, lr = ps_and_lrs.pop()
    print(f"{p=:.3f}, {lr=:.3e} popped with {device=}")
    process = Popen(make_args(p, lr), env={**environ, "CUDA_VISIBLE_DEVICES": str(device)})
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
    if len(ps) == 0 and all(process.poll() is not None for process in processes):
        break
    sleep(10)
