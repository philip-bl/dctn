from typing import Tuple

import torch
import torch.nn.functional as F


def score(model, dl, device) -> Tuple[float, float]:
    """Scores the model on all batches of dl. Which means it might skip only the last batch if
  dl.drop_last == True. Returns mean ce loss and accuracy."""
    num_samples = 0
    num_correct = 0
    sum_loss = 0.0
    with torch.no_grad():
        for x, y, _ in iter(dl):
            y = y.to(device)
            num_samples += len(y)
            unlogprobs = model(x.to(device))  # unnormalized log probabilities
            sum_loss += F.cross_entropy(unlogprobs, y, reduction="sum").item()
            num_correct += (unlogprobs.argmax(dim=1) == y).sum().item()
    loss = sum_loss / num_samples
    acc = num_correct / num_samples
    return loss, acc
