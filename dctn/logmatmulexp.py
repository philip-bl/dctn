import torch
from torch.utils.checkpoint import checkpoint


def logmatmulexp(log_A: torch.Tensor, log_B: torch.Tensor, /) -> torch.Tensor:
    """Given matrix log_A of shape ϴ×R and matrix log_B of shape R×I, calculates
  (log_A.exp() @ log_B.exp()).log() and its backward in a numerically stable way."""
    ϴ, R = log_A.shape
    I = log_B.shape[1]
    assert log_B.shape == (R, I)
    log_A_expanded = log_A.unsqueeze(2).expand((ϴ, R, I))
    log_B_expanded = log_B.unsqueeze(0).expand((ϴ, R, I))
    log_pairwise_products = log_A_expanded + log_B_expanded  # shape: (ϴ, R, I)
    return torch.logsumexp(log_pairwise_products, dim=1)


def logmatmulexp_lowmem(log_A: torch.Tensor, log_B: torch.Tensor, /) -> torch.Tensor:
    """Same as logmatmulexp, but doesn't save a (ϴ, R, I)-shaped tensor for backward pass.

  Given matrix log_A of shape ϴ×R and matrix log_B of shape R×I, calculates
  (log_A.exp() @ log_B.exp()).log() and its backward in a numerically stable way."""
    return checkpoint(logmatmulexp, log_A, log_B, preserve_rng_state=False)
