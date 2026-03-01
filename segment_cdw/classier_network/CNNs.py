from __future__ import annotations

import torch
from torch import nn
from torchvision import models
import timm


class ResNet50(nn.Module):
    """
    兼容你提供的 ResNet50 结构，去除对 opt 的依赖，使用显式参数。
    """

    def __init__(
        self,
        pretrained: bool = False,
        out_features: int = 1000,
        para_fusion: bool = False,
        n_handpara: int = 0,
    ) -> None:
        super().__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        in_channels = self.backbone.fc.in_features
        self.para_fusion = para_fusion
        self.n_handpara = n_handpara

        if self.para_fusion:
            self.backbone.fc = nn.Sequential(
                nn.Linear(in_channels + self.n_handpara, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, out_features),
            )
        else:
            self.backbone.fc = nn.Linear(in_channels, out_features)

    def forward(self, x: torch.Tensor, extre_para: torch.Tensor | None = None) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        if self.para_fusion:
            if extre_para is None:
                raise ValueError("para_fusion=True 需要提供 extre_para")
            x = torch.cat((x, extre_para), dim=1)

        x = self.backbone.fc(x)
        return x


class seresnext50_32x4d(nn.Module):
    def __init__(self, out_features, pretrained=True,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = timm.create_model('seresnext50_32x4d', pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, out_features)

    def forward(self, x):
        x = self.model(x)
        return x
    


