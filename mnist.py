import math
import logging
from functools import partial

from typing import Sequence, Any, Iterable, Optional, List

import click
import click_log

import torch
import torch.nn as nn
import torch.nn.functional as tnnf
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR

from ignite.metrics import Loss, Accuracy
from ignite.engine import Events
from ignite.contrib.handlers.param_scheduler import LRScheduler

import einops
from einops.layers.torch import Reduce as EinopsReduce

from libcrap import shuffled
from libcrap.torch import set_random_seeds
from libcrap.torch.click import (
    click_dataset_root_option,
    click_models_dir_option,
    click_tensorboard_log_dir_option,
    click_seed_and_device_options,
)
from libcrap.torch.training import (
    add_checkpointing,
    add_early_stopping,
    add_weights_and_grads_logging,
    setup_trainer,
    setup_evaluator,
    setup_tensorboard_logger,
    make_standard_prepare_batch_with_events,
    add_logging_input_images,
)

from dctn.conv_sbs import ManyConvSBS
from dctn.conv_sbs_spec import SBSSpecCore, Pos2D

logger = logging.getLogger()
click_log.basic_config(logger)

MNIST_DATASET_SIZE = 60000
NUM_LABELS = 10

MNIST_TRANSFORM = transforms.ToTensor()


h = 32
w = 32

h1 = 4
h2 = 4
hw = 4
w1 = 4
w2 = 4

o1 = 4
o2 = 4
o3 = 4
o4 = 4
o5 = 4

r1 = 8
r2 = 8
r3 = 8
r4 = 8


def permute_pixels(permutation: List[int], image: torch.Tensor) -> torch.Tensor:
    assert image.shape == (1, h, w)
    assert len(permutation) == h * w
    return image.reshape(h * w)[permutation].reshape(image.shape)


class DummyModel(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(1, 8, 3),
            nn.ELU(),
            nn.Conv2d(8, 32, 3),
            nn.ELU(),
            nn.Conv2d(32, 64, 3),
            nn.ELU(),
            nn.Conv2d(64, 10, 3),
            EinopsReduce("b c h w -> b c", "mean"),
        )


def batch_to_quantum(x: torch.Tensor) -> torch.Tensor:
    batch = einops.rearrange(x, "b () h w -> b h w")
    batch_quantum = torch.stack((torch.sin(batch), torch.cos(batch)), dim=3)
    assert batch_quantum.shape == (*batch.shape, 2)  # b h w 2
    return batch_quantum


class DCTNMnistModel(nn.Module):
    def __init__(self, num_sbs_layers: int, bond_dim_size: int, trace_edge: bool):
        super().__init__()
        assert num_sbs_layers >= 2
        cores_specs = (
            (
                SBSSpecCore(Pos2D(-1, -1), 1),
                SBSSpecCore(Pos2D(-1, 0), 1),
                SBSSpecCore(Pos2D(-1, 1), 1),
                SBSSpecCore(Pos2D(0, 1), 1),
                SBSSpecCore(Pos2D(0, 0), 2),
                SBSSpecCore(Pos2D(0, -1), 1),
                SBSSpecCore(Pos2D(1, -1), 1),
                SBSSpecCore(Pos2D(1, 0), 1),
                SBSSpecCore(Pos2D(1, 1), 1),
            ),
            (
                SBSSpecCore(Pos2D(-1, -1), 1),
                SBSSpecCore(Pos2D(0, -1), 1),
                SBSSpecCore(Pos2D(1, -1), 1),
                SBSSpecCore(Pos2D(1, 0), 1),
                SBSSpecCore(Pos2D(0, 0), 2),
                SBSSpecCore(Pos2D(-1, 0), 1),
                SBSSpecCore(Pos2D(-1, 1), 1),
                SBSSpecCore(Pos2D(0, 1), 1),
                SBSSpecCore(Pos2D(1, 1), 1),
            ),
        )
        final_string_cores_spec = (
            SBSSpecCore(Pos2D(-1, -1), 1),
            SBSSpecCore(Pos2D(-1, 0), 1),
            SBSSpecCore(Pos2D(-1, 1), 1),
            SBSSpecCore(Pos2D(0, 1), 1),
            SBSSpecCore(Pos2D(0, 0), NUM_LABELS),
            SBSSpecCore(Pos2D(0, -1), 1),
            SBSSpecCore(Pos2D(1, -1), 1),
            SBSSpecCore(Pos2D(1, 0), 1),
            SBSSpecCore(Pos2D(1, 1), 1),
        )
        self.conv_sbses = nn.Sequential(
            ManyConvSBS(
                in_num_channels=1,
                in_quantum_dim_size=2,
                bond_dim_size=bond_dim_size,
                trace_edge=trace_edge,
                cores_specs=cores_specs,
            ),
            *(
                ManyConvSBS(
                    in_num_channels=2,
                    in_quantum_dim_size=2,
                    bond_dim_size=bond_dim_size,
                    trace_edge=trace_edge,
                    cores_specs=cores_specs,
                )
                for i in range(num_sbs_layers - 2)
            ),
            ManyConvSBS(
                in_num_channels=2,
                in_quantum_dim_size=2,
                bond_dim_size=bond_dim_size,
                trace_edge=trace_edge,
                cores_specs=(final_string_cores_spec,),
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        intermediate = (batch_to_quantum(x),)
        for conv_sbs in self.conv_sbses:
            intermediate = conv_sbs(intermediate)
        (result,) = intermediate
        return einops.reduce(result, "b h w l -> b l", "mean")


@click.command()
@click_log.simple_verbosity_option(logger)
@click_dataset_root_option()
@click_models_dir_option()
@click_tensorboard_log_dir_option()
@click.option(
    "--train-dataset-size",
    "-t",
    type=click.IntRange(1, MNIST_DATASET_SIZE),
    default=58000,
)
@click.option("--learning-rate", "-r", type=float, default=1e-2)
@click.option("--batch-size", "-b", type=int, default=100)
@click.option("--shuffle-pixels", is_flag=True)
@click_seed_and_device_options(default_device="cpu")
def main(
    dataset_root,
    train_dataset_size,
    tb_log_dir,
    models_dir,
    learning_rate,
    batch_size,
    device,
    seed,
    shuffle_pixels,
):
    if not shuffle_pixels:
        transform = MNIST_TRANSFORM
    else:
        print("Pixel shuffling is enabled")
        pixel_shuffle_transform = transforms.Lambda(
            partial(permute_pixels, shuffled(range(h * w)))
        )
        transform = transforms.Compose((MNIST_TRANSFORM, pixel_shuffle_transform))
    dataset = MNIST(dataset_root, train=True, download=True, transform=transform)
    assert len(dataset) == MNIST_DATASET_SIZE
    train_dataset, val_dataset = random_split(
        dataset, (train_dataset_size, MNIST_DATASET_SIZE - train_dataset_size)
    )
    train_loader, val_loader = (
        DataLoader(
            dataset_,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=(device.type == "cuda"),
        )
        for dataset_ in (train_dataset, val_dataset)
    )
    model = DCTNMnistModel(5, 6, False)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    prepare_batch_for_trainer = make_standard_prepare_batch_with_events(device)
    trainer = setup_trainer(
        model,
        optimizer,
        tnnf.cross_entropy,
        device=device,
        prepare_batch=prepare_batch_for_trainer,
    )
    scheduler = LRScheduler(
        torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8547)
    )
    trainer.add_event_handler(Events.EPOCH_STARTED, scheduler)
    metrics = {"cross_entropy_loss": Loss(tnnf.cross_entropy), "accuracy": Accuracy()}
    prepare_batch_for_val_evaluator = make_standard_prepare_batch_with_events(device)
    val_evaluator = setup_evaluator(
        model,
        trainer,
        val_loader,
        metrics,
        device=device,
        prepare_batch=prepare_batch_for_val_evaluator,
    )
    add_checkpointing(
        models_dir,
        "cross_entropy_loss",
        val_evaluator,
        objects_to_save={"model": model},
        model=model,
    )
    add_early_stopping(
        trainer, val_evaluator, "cross_entropy_loss", patience_num_evaluations=25
    )
    with setup_tensorboard_logger(
        tb_log_dir, trainer, metrics.keys(), {"val": val_evaluator}, model=model
    ) as tb_logger:
        add_weights_and_grads_logging(trainer, tb_logger, model)
        add_logging_input_images(tb_logger, trainer, "train", prepare_batch_for_trainer)
        add_logging_input_images(
            tb_logger,
            val_evaluator,
            "val",
            prepare_batch_for_val_evaluator,
            another_engine=trainer,
        )
        trainer.run(train_loader, max_epochs=100)


if __name__ == "__main__":
    main()
