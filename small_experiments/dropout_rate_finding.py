from random import shuffle, seed
from os import environ
from os.path import expanduser
from subprocess import Popen
from time import sleep
from typing import Tuple, Any

import numpy as np

min_p = 0.01
max_p = 1.0
num_experiments = 20

seed(0)
ps = list(np.linspace(min_p, max_p, num=num_experiments))
shuffle(ps)
assert len(ps) >= 2

common_args = (
    "python",
    expanduser("~/projects/dctn/new_runner.py"),
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
    "--lr",
    "1.821e-4",
    "--patience",
    "13",
    "--reg-type",
    "epses_composition",
    "--reg-coeff",
    "1e-2",
    "--no-es-train-acc",
    "--no-es-train-mean-ce",
    "--no-breakpoint-on-nan-loss",
    "--init-epses-composition-unit-empirical-output-std",
)


def make_args(p: float) -> Tuple[str, ...]:
    return (*common_args, "--dropout-p", str(p))


def create_process(device: int):
    p = ps.pop()
    print(f"{p=} popped with {device=}")
    process = Popen(make_args(p), env={**environ, "CUDA_VISIBLE_DEVICES": str(device)})
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
