from typing import *
from subprocess import run
import os.path
from os.path import join
import logging
from math import pi

import click

import torch
from torch.optim import Adam, SGD
import torch.nn.functional as F

from libcrap import get_now_as_str, save_json
from libcrap.torch import set_random_seeds

from dctn.eps_plus_linear import EPSPlusLinear
from dctn.evaluation import score
from dctn.dataset_loading import get_fashionmnist_data_loaders, get_mnist_data_loaders
from dctn.training import (
  train, every_n_iters_intervals, LastModelsCheckpointer, BestModelCheckpointer, ValuesNotImprovingEarlyStopper)

DIFF_FNAME = "git_diff_with_HEAD.patch"
RUN_INFO_FNAME = "run_info.txt"
LOG_FNAME = "log.log"

def save_git_diff_with_head(fname: str) -> None:
  assert "dctn" in os.getcwd()
  diff: bytes = run(("git", "diff", "HEAD"), capture_output=True, check=True).stdout
  with open(fname, "wb") as diff_f:
    diff_f.write(diff)

def get_git_commit_info() -> str:
  return run(("git", "show", "--format=oneline"), text=True, capture_output=True, check=True) \
    .stdout.split("\n")[0]


@click.command()
@click.option("--experiments-dir", type=click.Path(file_okay=False), required=True)
@click.option("--ds-type", type=click.Choice(("mnist","fashionmnist"), case_sensitive=False))
@click.option("--ds-path", type=click.Path(exists=True, file_okay=False))
@click.option("--device", type=str, default="cuda")
@click.option("--seed", type=int, default=0)
@click.option("-v", "--verbosity", default="INFO",
  type=lambda s: {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARN": logging.WARN,
                  "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL}[s])
@click.option("--kernel-size", type=int, required=True)
@click.option("--out-size", type=int, required=True)
@click.option("--batch-size", type=int, required=True)
@click.option("--load-model-state", type=click.Path(exists=True, dir_okay=False))
@click.option("--optimizer", type=click.Choice(("adam", "sgd"), case_sensitive=False))
@click.option("--lr", type=float)
@click.option("--wd", type=float, help="weight decay", default=0.)
@click.option("--patience", type=int, help="early stopping patience num evaluations", default=20)
@click.option("--keep-last-models", type=int, help="how many last models to keep", default=10)
@click.option("--old-scaling/--no-old-scaling", default=False)
def main(**kwargs) -> None:
  kwargs["output_dir"] = join(kwargs["experiments_dir"], get_now_as_str(False, True, True))
  assert not os.path.exists(kwargs["output_dir"])
  os.mkdir(kwargs["output_dir"])
  save_json({**kwargs, "commit": get_git_commit_info()},
            join(kwargs["output_dir"], RUN_INFO_FNAME))
  save_git_diff_with_head(join(kwargs["output_dir"], DIFF_FNAME))
  kwargs["device"] = torch.device(kwargs["device"])
  
  logging.basicConfig(
    level=kwargs["verbosity"], handlers=(
      logging.StreamHandler(),
      logging.FileHandler(join(kwargs["output_dir"], LOG_FNAME), "w", "utf-8")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", force=True)
  logger = logging.getLogger(__name__)
  logger.info(f"{kwargs['output_dir']=}")

  dev = kwargs["device"]
  set_random_seeds(dev, kwargs["seed"])
  model = EPSPlusLinear(kwargs["kernel_size"], kwargs["out_size"]).to(dev)
  if kwargs["old_scaling"]:
    model[0].core.data = torch.randn_like(model[0].core) / 4
    model[-1].weight.data = torch.rand_like(model[-1].weight) * 0.04 - 0.02
  if kwargs["load_model_state"] is not None:
    model.load_state_dict(torch.load(kwargs["load_model_state"], dev))
  get_dls = {"mnist": get_mnist_data_loaders, "fashionmnist": get_fashionmnist_data_loaders}[
    kwargs["ds_type"]]
  if not kwargs["old_scaling"]:
    train_dl, val_dl, test_dl = get_dls(
      kwargs["ds_path"], kwargs["batch_size"], dev, autoscale_kernel_size=kwargs["kernel_size"])
  else:
    train_dl, val_dl, test_dl = get_dls(
      kwargs["ds_path"], kwargs["batch_size"], dev, (lambda X: (X*pi/2.).sin()**2/2, lambda X: (X*pi/2.).cos()**2/2))

  eval_schedule = every_n_iters_intervals((10, 1), (100, 10), (1000, 100), (20000, 1000), (None, 10000))

  @eval_schedule
  def evaluate_and_log(st_x, st_it):
    st_x["model"].eval()
    st_it["train_mean_ce"], st_it["train_acc"] = score(st_x["model"], train_dl, st_x["dev"])
    st_it["val_mean_ce"], st_it["val_acc"] = score(st_x["model"], val_dl, st_x["dev"])
    logger.info(f"After {st_it['num_iters_done']:07} iters: "
                f"train/val mean_ce={st_it['train_mean_ce']:.5f}/{st_it['val_mean_ce']:.5f} "
                f"acc={st_it['train_acc']:.2%}/{st_it['val_acc']:.2%}")

  last_models_checkpointer = eval_schedule(LastModelsCheckpointer(
    kwargs["output_dir"], kwargs["keep_last_models"]))
  metrics = (("train_acc", False), ("val_acc", False), ("train_mean_ce", True), ("val_mean_ce", True))
  best_value_checkpointers = tuple(
    eval_schedule(BestModelCheckpointer(kwargs["output_dir"], *metric)) for metric in metrics)
  early_stopper = eval_schedule(ValuesNotImprovingEarlyStopper(kwargs["patience"], metrics))
  optimizer = {"adam": Adam, "sgd": SGD}[kwargs["optimizer"]](
    model.parameters(), kwargs["lr"], weight_decay=kwargs["wd"])
  set_random_seeds(dev, kwargs["seed"])
  st_x, st_it = train(
    train_dl, model, optimizer, kwargs["device"], F.cross_entropy,
    [evaluate_and_log, last_models_checkpointer, *best_value_checkpointers, early_stopper], [], [])


if __name__ == "__main__":
  main()
