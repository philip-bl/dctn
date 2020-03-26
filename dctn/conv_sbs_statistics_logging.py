import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from ignite.engine import Engine, Events
from ignite.contrib.handlers import CustomPeriodicEvent

from .conv_sbs import ConvSBS


def add_conv_sbs_tt_tensor_statistics_logging(
  model: nn.Module, writer: SummaryWriter, trainer: Engine, every_n_iters: int = 1
) -> None:
  """Adds logging of mean and std of the tt tensor of ConvSBS to tensorboard. Of the actual
  TT tensor, not of some dumb array."""
  cpe = CustomPeriodicEvent(n_iterations=every_n_iters)
  cpe.attach(trainer)
  event = getattr(cpe.Events, f"ITERATIONS_{every_n_iters}_STARTED",)
  def add_for_one_conv_sbs(module: ConvSBS, module_name: str) -> None:
    @trainer.on(event)
    def log(trainer) -> None:
      writer.add_scalar(
        f"mean_of_tt_tensor/{module_name}",
        module.mean(),
        trainer.state.iteration,
      )
      writer.add_scalar(
        f"std_of_tt_tensor/{module_name}",
        module.var() ** 0.5,
        trainer.state.iteration,
      )
  for module_name, module in model.named_modules():
    if isinstance(module, ConvSBS):
      add_for_one_conv_sbs(module, module_name)

      # TODO also add logging the same for gradient. actually, can I even do that?
      # does that even make sense?
