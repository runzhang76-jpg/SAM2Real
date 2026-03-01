"""学生网络的头部抽象。"""

from __future__ import annotations

from typing import Any

try:
    import torch.nn as nn
except Exception:  # pragma: no cover - 可选依赖
    nn = None  # type: ignore


class MaskHead(nn.Module):
    """轻量 mask head 实现。"""

    def __init__(self, in_channels: int = 32, out_channels: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, features: Any) -> Any:
        return self.block(features)
