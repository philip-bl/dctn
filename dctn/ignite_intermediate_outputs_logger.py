from typing import Callable

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from ignite.engine import Events
from ignite.contrib.handlers import CustomPeriodicEvent

from .base_intermediate_outputs_logger import (
    LoggerTransformType,
    SimpleIntermediateOutputsLogger,
    log_dumb_max_of_abs,
    log_dumb_mean_of_abs,
    log_dumb_min_of_abs,
)


def create_every_n_iters_intermediate_outputs_logger(
    model: nn.Module,
    writer: SummaryWriter,
    module_filter: Callable[[str, nn.Module], bool],
    engine,
    tag_prefix,  # e.g. train or val
    every_n_iters: int = 1,
    loggers=(log_dumb_mean_of_abs, log_dumb_min_of_abs, log_dumb_max_of_abs),
    use_input: bool = False,
) -> None:
    siol = SimpleIntermediateOutputsLogger(model, writer, module_filter, loggers, use_input)
    siol.tag_prefix = f"{tag_prefix}_intermediate_output"
    cpe = CustomPeriodicEvent(n_iterations=every_n_iters)
    cpe.attach(engine)

    @engine.on(getattr(cpe.Events, f"ITERATIONS_{every_n_iters}_STARTED"))
    def enable(engine) -> None:
        siol.enabled = True
        siol.step = engine.state.iteration

    @engine.on(Events.ITERATION_COMPLETED)
    def disable(engine) -> None:
        siol.enabled = False
