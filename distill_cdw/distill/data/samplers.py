"""可靠性感知蒸馏的采样策略。"""

from __future__ import annotations

import random
from typing import Any, Iterable, Iterator, List, Optional

from distill.core.structures import PseudoLabelInstance

try:
    import torch
    from torch.utils.data import Sampler
except Exception:  # pragma: no cover - torch 可选
    torch = None
    Sampler = object  # type: ignore


def _mean_reliability(instances: List[PseudoLabelInstance]) -> float:
    if not instances:
        return 0.0
    return float(sum(inst.reliability for inst in instances) / max(1, len(instances)))


def _extract_reliabilities(dataset: Any) -> List[float]:
    reliabilities: List[float] = []
    if hasattr(dataset, "instances_by_image") and hasattr(dataset, "image_index"):
        for image_info in dataset.image_index:
            image_id = int(image_info.get("id", 0))
            insts = dataset.instances_by_image.get(image_id, [])
            reliabilities.append(_mean_reliability(insts))
        return reliabilities

    if hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
        for idx in range(len(dataset)):
            sample = dataset[idx]
            insts = sample.get("instances", []) if isinstance(sample, dict) else []
            reliabilities.append(_mean_reliability(insts))
        return reliabilities

    return []


class ReliabilitySampler(Sampler[int]):
    """按伪标签可靠性进行筛选与打乱的采样器。"""

    def __init__(
        self,
        dataset: Any,
        min_reliability: float = 0.0,
        max_reliability: float = 1.0,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.min_reliability = float(min_reliability)
        self.max_reliability = float(max_reliability)
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.reliabilities = _extract_reliabilities(dataset)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_reliability_range(self, min_reliability: float, max_reliability: float) -> None:
        self.min_reliability = float(min_reliability)
        self.max_reliability = float(max_reliability)

    def __iter__(self) -> Iterator[int]:
        indices = [
            idx
            for idx, rel in enumerate(self.reliabilities)
            if self.min_reliability <= rel <= self.max_reliability
        ]
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(indices)
        for idx in indices:
            yield idx

    def __len__(self) -> int:
        return len(self.reliabilities)
