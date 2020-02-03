import enum
from functools import partial

from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class RecordType(enum.Enum):
    SCALAR = enum.auto()
    HISTOGRAM = enum.auto()


LoggerTransformType = Tuple[str, RecordType, Callable[[torch.Tensor], torch.Tensor]]
# (name, type of logging e.g. scalar or histogram, the transforming function which takes output and returns a scalar)

log_dumb_mean_of_abs: LoggerTransformType = (
    "dumb_mean_of_abs",
    RecordType.SCALAR,
    lambda x: x.abs().mean(),
)
log_dumb_max_of_abs: LoggerTransformType = (
    "dumb_max_of_abs",
    RecordType.SCALAR,
    lambda x: x.abs().max(),
)
log_dumb_min_of_abs: LoggerTransformType = (
    "dumb_min_of_abs",
    RecordType.SCALAR,
    lambda x: x.abs().min(),
)


class SimpleIntermediateOutputsLogger:
    """How to use:
    1 Initialize - it will attach forward hooks
    2 When you want the logging to actually happen, set tag_prefix, step and enabled
    3 After you've done the forward you wanted to log set enabled=False"""

    def __init__(
        self,
        model: nn.Module,
        writer: SummaryWriter,
        module_filter: Callable[[str, nn.Module], bool],
        loggers: Tuple[LoggerTransformType, ...] = (
            log_dumb_mean_of_abs,
            log_dumb_min_of_abs,
            log_dumb_max_of_abs,
        ),
    ):
        self.enabled = False
        self.tag_prefix = None
        self.step = None

        def hook(module_name, module, input_, output) -> None:
            if self.enabled:
                for logger_name, record_type, logger_transform in loggers:
                    tag = f"{self.tag_prefix}_{logger_name}/{module_name}"
                    if record_type == RecordType.SCALAR:
                        writer.add_scalar(
                            tag, logger_transform(output), self.step,
                        )
                    elif record_type == RecordType.HISTOGRAM:
                        writer.add_histogram(tag, logger_transform(output), self.step)
                    else:
                        assert "This should never happen"

        self._hooks_handles = tuple(
            module.register_forward_hook(partial(hook, module_name))
            for (module_name, module) in model.named_modules()
            if module_filter(module_name, module)
        )

    def remove_hooks(self) -> None:
        for handle in self._hooks_handles:
            handle.remove()
