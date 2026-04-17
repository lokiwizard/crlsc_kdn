from __future__ import annotations

import timm
import torch
from torch import nn


class TimmEncoder(nn.Module):
    def __init__(self, model_name: str = "resnet18", pretrained: bool = False) -> None:
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.network = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.feature_dim = getattr(self.network, "num_features")
        if self.feature_dim is None:
            raise ValueError(f"Could not resolve feature_dim for timm model: {model_name}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)
