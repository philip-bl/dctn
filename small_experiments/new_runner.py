from typing import *
from subprocess import run
import os.path
from os.path import join
import logging

import click

import torch

from libcrap import get_now_as_str, save_json
from libcrap.torch import set_random_seeds

from dctn.eps_plus_linear import EPSPlusLinear
from dctn.evaluation import score
from dctn.dataset_loading import get_fashionmnist_data_loaders, get_mnist_data_loaders

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
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
  logger = logging.getLogger(__name__)
  logger.info(f"{kwargs['output_dir']=}")

  dev = kwargs["device"]
  model = EPSPlusLinear(kwargs["kernel_size"], kwargs["out_size"]).to(dev)
  if kwargs["load_model_state"] is not None:
    model.load_state_dict(torch.load(kwargs["load_model_state"], dev))
  get_dls = {"mnist": get_mnist_data_loaders, "fashionmnist": get_fashionmnist_data_loaders}[
    kwargs["ds_type"]]
  train_dl, val_dl, test_dl = get_dls(kwargs["ds_path"], kwargs["batch_size"], dev)
  set_random_seeds(dev, kwargs["seed"])
  logger.info("On train loss={0:.4f}, acc={1:.2%}".format(*score(model, train_dl, dev)))
  logger.info("On val   loss={0:.4f}, acc={1:.2%}".format(*score(model, val_dl, dev)))
  logger.info("On test  loss={0:.4f}, acc={1:.2%}".format(*score(model, test_dl, dev)))


if __name__ == "__main__":
  main()
