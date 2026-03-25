""""""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from sam2real.core.structures import InstancePrediction

import torch
import torch.nn as nn



class StudentModel(nn.Module, ABC):
    """"""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, images: "torch.Tensor", targets: Optional[Any] = None) -> Dict[str, Any]:
        """"""

    @abstractmethod
    def predict(self, images: Any, **kwargs: Any) -> List[List[InstancePrediction]]:
        """"""

    @abstractmethod
    def load_weights(self, path: str) -> None:
        """"""

    @abstractmethod
    def save_weights(self, path: str) -> None:
        """"""

    def export(self, *args: Any, **kwargs: Any) -> None:
        """"""

        raise NotImplementedError(" export")
