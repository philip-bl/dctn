import logging
from functools import wraps
from collections import deque
import os
from typing import Iterator, Any, Tuple, Dict, Callable

import torch
from torch.utils.data import DataLoader

from more_itertools import last


def batches_forever(dl: DataLoader) -> Iterator[Any]:
    while True:
        yield from iter(dl)


StX = Dict[Any, Any]
StIt = Dict[Any, Any]


def train(
    dl: DataLoader, model, optimizer, dev, loss_fn, at_iter_start, after_back, after_param_upd
) -> Tuple[StX, StIt]:
    """`loss_fn` must take (model_output, y) and return the loss as a 0-dim torch.tensor.
  `at_iter_start` - functions which will run each time when an iter starts.
  `after_back` - functions which will run each time after `st_x["model"].backward()`.
  `after_param_upd` - functions which will run each time after `st_x["optimizer"].step()`."""
    st_x = {
        "model": model.to(dev),
        "optimizer": optimizer,
        "loss_fn": loss_fn,
        "at_iter_start": list(at_iter_start),
        "after_back": list(after_back),
        "after_param_upd": list(after_param_upd),
        "dev": dev,
    }
    # st_x stands for state across iterations, i.e. state which remains between iterations
    del model, optimizer, dev, loss_fn, after_back  # I don't want to accidentally use these

    def run_x_funcs(key):
        for f in st_x[key][:]:
            f(st_x, st_it)

    for num_iters_done, (x, y, indices) in enumerate(batches_forever(dl)):
        st_it = {
            "num_iters_done": num_iters_done,
            "x": x.to(st_x["dev"]),
            "y": y.to(st_x["dev"]),
            "indices": indices.to(st_x["dev"]),
            "stop": False,
        }
        # st_it stands for state local to this iteration
        del num_iters_done, x, y, indices
        run_x_funcs("at_iter_start")
        st_x["model"].train()
        st_it["output"] = st_x["model"](st_it["x"])
        st_it["loss"] = st_x["loss_fn"](st_it["output"], st_it["y"])
        st_x["optimizer"].zero_grad()
        st_it["loss"].backward()
        run_x_funcs("after_back")
        st_x["optimizer"].step()
        run_x_funcs("after_param_upd")
        if st_it["stop"]:
            break
    return st_x, st_it


def every_n_iters_intervals(*intervals):
    """Each interval must be a pair of two ints. The first int is the interval's length, the second int
  is the frequency of invocation during that interval. The last interval's length may be None meaning forever."""
    intervals = list(intervals)
    if intervals[-1][0] is not None:
        intervals.append(None, 1)
    intervals_starts = [0]
    for length, _ in intervals[:-1]:
        intervals_starts.append(intervals_starts[-1] + length)
    assert len(intervals_starts) == len(intervals)

    def decorate(func: Callable[[StX, StIt], None]) -> Callable[[StX, StIt], None]:
        def wrapped_func(st_x: StX, st_it: StIt) -> None:
            freq = last(
                freq
                for start, (_, freq) in zip(intervals_starts, intervals)
                if st_it["num_iters_done"] >= start
            )
            if st_it["num_iters_done"] % freq == 0:
                func(st_x, st_it)

        return wrapped_func

    return decorate


class Checkpointer:
    def __init__(self, dir: str):
        self.dir = dir

    def save(self, st_x: StX, filename: str) -> None:
        torch.save(st_x["model"].state_dict(), os.path.join(self.dir, filename))

    def remove_file(self, filename: str) -> None:
        os.remove(os.path.join(self.dir, filename))


class LastModelsCheckpointer(Checkpointer):
    def __init__(self, dir: str, n: int):
        """Keeps `n` checkpoints of last `n` models (one per call to an instance of this class) in `dir`."""
        super().__init__(dir)
        assert n >= 1
        self.n = n
        self.filenames = deque()

    def __call__(self, st_x: StX, st_it: StIt):
        nitd = st_it["num_iters_done"]
        tracc = st_it["train_acc"]
        vacc = st_it["val_acc"]
        trmce = st_it["train_mean_ce"]
        vmce = st_it["val_mean_ce"]
        filename = f"model_{nitd=:07}_{tracc=:.4f}_{vacc=:.4f}_{trmce=:.4f}_{vmce=:.4f}.pth"
        self.save(st_x, filename)
        self.filenames.appendleft(filename)
        while len(self.filenames) > self.n:
            self.remove_file(self.filenames.pop())


class BestModelCheckpointer(Checkpointer):
    def __init__(self, dir: str, key: str, low_is_good: bool):
        super().__init__(dir)
        self.key = key
        self.low_is_good = low_is_good
        self.best_value = float("+inf") if low_is_good else float("-inf")
        self.filename = None

    def __call__(self, st_x: StX, st_it: StIt):
        value = st_it[self.key]
        if (
            self.low_is_good
            and value < self.best_value
            or not self.low_is_good
            and value > self.best_value
        ):
            nitd = st_it["num_iters_done"]
            tracc = st_it["train_acc"]
            vacc = st_it["val_acc"]
            trmce = st_it["train_mean_ce"]
            vmce = st_it["val_mean_ce"]
            new_filename = f"model_best_{self.key}_{nitd=:07}_{tracc=:.4f}_{vacc=:.4f}_{trmce=:.4f}_{vmce=:.4f}.pth"
            self.save(st_x, new_filename)
            self.best_value = value
            if self.filename is not None:
                self.remove_file(self.filename)
            self.filename = new_filename


class ValuesNotImprovingEarlyStopper:
    def __init__(self, patience: int, keys: Tuple[Tuple[str, bool], ...]):
        """If no value improves in `patience` calls to an instance of this object, stop the iterations.
    Each element of `values` must be a tuple of form (key, low_is_good)."""
        self.keys = keys
        self.best_values = [
            float("+inf") if low_is_good else float("-inf") for _, low_is_good in keys
        ]
        self.num_bad_calls = 0
        self.patience = patience

    def __call__(self, st_x: StX, st_it: StIt):
        improvement = False
        for i, (key, low_is_good) in enumerate(self.keys):
            value = st_it[key]
            best_value = self.best_values[i]
            if low_is_good and value < best_value or not low_is_good and value > best_value:
                self.best_values[i] = value
                improvement = True
        if improvement:
            self.num_bad_calls = 0
        else:
            self.num_bad_calls += 1
        if self.num_bad_calls > self.patience:
            st_it["stop"] = True
            logging.getLogger(__name__).info(f"Early stopping at {st_it['num_iters_done']=}")
