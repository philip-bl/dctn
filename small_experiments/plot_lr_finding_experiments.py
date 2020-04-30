import os
from typing import Dict, Tuple, List, Callable
import re
from more_itertools import first_true
from libcrap import traverse_files_no_recursion, load_json, get_now_as_str
import pandas as pd
from bokeh.plotting import figure, output_file, show
from bokeh.io import save

DIR = "/mnt/important/experiments/2_epses_plus_linear_fashionmnist/making_empirical_std_one_lr_finding"

subdirs = tuple(entry.path for entry in os.scandir(DIR) if entry.is_dir())

fname_pattern = r"model_best_val_acc_nitd=(\d+)_tracc=\d+\.\d+_vacc=(\d+\.\d+)_trmce=\d+\.\d+_vmce=\d+\.\d+\.pth$"


def get_best_vacc_info(subdir: str) -> Tuple[int, float]:
    for fname in traverse_files_no_recursion(subdir, ("pth",)):
        if match := re.search(fname_pattern, fname,):
            nitd, vacc = match.groups()
            return int(nitd), float(vacc)
    assert False


def get_lr(subdir: str) -> float:
    json = load_json(os.path.join(subdir, "run_info.txt"))
    return json["lr"]


d = {"lr": [], "best_vacc": [], "best_vacc_nitd": []}
for subdir in subdirs:
    d["lr"].append(get_lr(subdir))
    best_vacc_nitd, best_vacc = get_best_vacc_info(subdir)
    d["best_vacc_nitd"].append(best_vacc_nitd)
    d["best_vacc"].append(best_vacc)

df = pd.DataFrame.from_dict(d).set_index("lr").sort_index()

now = get_now_as_str(utc=False, year=True)
key = "best_vacc"
p = figure(title=key, x_axis_type="log")
p.line(df.index, df[key])
p_fname = os.path.join(DIR, f"{now}_{key}.html")
output_file(p_fname, title=key, mode="inline")
save(p, p_fname, title=key)


key = "best_vacc_nitd"
p = figure(title=key, x_axis_type="log", y_axis_type="log")
p.line(df.index, df[key])
p_fname = os.path.join(DIR, f"{now}_{key}.html")
output_file(p_fname, title=key, mode="inline")
save(p, p_fname, title=key)
