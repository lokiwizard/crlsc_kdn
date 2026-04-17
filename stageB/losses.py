from __future__ import annotations

import torch
import torch.nn.functional as F


def nt_xent_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.2,
) -> tuple[torch.Tensor, float]:
    if z1.shape != z2.shape:
        raise ValueError(f"Mismatched shapes: {tuple(z1.shape)} vs {tuple(z2.shape)}")

    batch_size = z1.shape[0]
    if batch_size < 2:
        raise ValueError("SimCLR requires batch_size >= 2")

    representations = torch.cat([z1, z2], dim=0)
    representations = F.normalize(representations, dim=-1)

    logits = representations @ representations.T
    logits = logits / temperature

    diagonal_mask = torch.eye(2 * batch_size, device=logits.device, dtype=torch.bool)
    logits = logits.masked_fill(diagonal_mask, float("-inf"))

    targets = torch.arange(batch_size, device=logits.device)
    targets = torch.cat([targets + batch_size, targets], dim=0)

    loss = F.cross_entropy(logits, targets)
    predictions = logits.argmax(dim=1)
    accuracy = (predictions == targets).float().mean().item()
    return loss, accuracy
