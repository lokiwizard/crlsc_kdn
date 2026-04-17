from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18
from torchvision.models import resnet34


class ResNetEncoder(nn.Module):
    def __init__(self, backbone: str = "resnet18") -> None:
        super().__init__()

        builders = {
            "resnet18": (resnet18, 512),
            "resnet34": (resnet34, 512),
        }
        if backbone not in builders:
            raise ValueError(f"Unsupported backbone: {backbone}")

        builder, feature_dim = builders[backbone]
        network = builder(weights=None)
        network.fc = nn.Identity()

        self.backbone_name = backbone
        self.feature_dim = feature_dim
        self.network = network

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)
