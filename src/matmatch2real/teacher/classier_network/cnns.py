"""Classifier model factory functions."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNeXt50_32X4D_Weights, resnext50_32x4d


def seresnext50_32x4d(pretrained: bool = False, out_features: int = 1000) -> nn.Module:
    weights = ResNeXt50_32X4D_Weights.DEFAULT if pretrained else None
    model = resnext50_32x4d(weights=weights)
    in_features = int(model.fc.in_features)
    model.fc = nn.Linear(in_features, out_features)
    return model
