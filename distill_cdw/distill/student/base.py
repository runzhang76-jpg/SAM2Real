"""学生模型统一接口定义。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from distill.core.structures import InstancePrediction

import torch
import torch.nn as nn



class StudentModel(nn.Module, ABC):
    """学生模型抽象基类。"""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, images: "torch.Tensor", targets: Optional[Any] = None) -> Dict[str, Any]:
        """前向过程，训练时可返回监督损失与中间表示。"""

    @abstractmethod
    def predict(self, images: Any, **kwargs: Any) -> List[List[InstancePrediction]]:
        """推理接口，返回统一的实例预测结果。"""

    @abstractmethod
    def load_weights(self, path: str) -> None:
        """加载权重。"""

    @abstractmethod
    def save_weights(self, path: str) -> None:
        """保存权重。"""

    def export(self, *args: Any, **kwargs: Any) -> None:
        """模型导出接口（可选扩展）。"""

        raise NotImplementedError("该学生模型尚未实现 export。")
