from __future__ import annotations

import torch
from torch import nn


class AlignmentHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim is None:
            if in_dim == out_dim:
                self.layers = nn.Identity()
            else:
                self.layers = nn.Linear(in_dim, out_dim)
        else:
            self.layers = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)
