import os.path
from typing import Tuple, Optional, Dict, Any, List
from functools import reduce
from pprint import pformat


import click

from libcrap import load_json, traverse_files, save_json, shuffled

import plot_training


@click.command()
@click.argument(
    "experiments-dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, writable=False),
)
@click.argument(
    "output-json-path",
    type=click.Path(exists=False, dir_okay=False, file_okay=True, writable=True),
)
@click.option("-k", "--keys-of-interest", type=str, multiple=True)
@click.option(
    "-r", "--random-subset-of-size", type=int, default=25, help="Set -1 to use all experiments"
)
def main(
    experiments_dir: str, output_json_path: str, keys_of_interest, random_subset_of_size: int
) -> None:
    runs_infos_paths = tuple(
        path for path in traverse_files(experiments_dir) if path.endswith("run_info.txt")
    )
    if random_subset_of_size != -1:
        runs_infos_paths = tuple(shuffled(runs_infos_paths)[:random_subset_of_size])
    experiments_dirs_relpaths = tuple(
        os.path.relpath(os.path.dirname(path), experiments_dir) for path in runs_infos_paths
    )  # contains relative paths to each dir containing an experiment in `experiments_dir`
    runs_infos: Dict[str, Any] = tuple(load_json(path) for path in runs_infos_paths)
    union_of_keys = reduce(
        lambda x, y: x | y, (frozenset(run_info.keys()) for run_info in runs_infos)
    )
    assert union_of_keys.issuperset(keys_of_interest)
    shared_items = {}
    for key in union_of_keys:
        if key not in runs_infos[0]:
            continue
        value = runs_infos[0][key]
        if all(key in run_info and run_info[key] == value for run_info in runs_infos):
            shared_items[key] = value
    non_shared_items = tuple(
        {k: v for k, v in run_info.items() if k not in shared_items} for run_info in runs_infos
    )
    names = (
        tuple(str(i) for i in range(len(runs_infos)))
        if not keys_of_interest
        else tuple(
            str({k: v for k, v in d.items() if k in keys_of_interest})
            for d in non_shared_items
        )
    )
    descriptions = tuple(str(d) for d in non_shared_items)
    json_struct = {
        "common_description": pformat(shared_items, indent=0),
        "experiments": [
            {"rel_dir": rel_dir, "name": name, "description": description}
            for (rel_dir, name, description) in zip(
                experiments_dirs_relpaths, names, descriptions
            )
        ],
    }
    save_json(json_struct, output_json_path)
