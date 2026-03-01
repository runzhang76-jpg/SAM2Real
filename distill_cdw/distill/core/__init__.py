"""核心训练循环与框架工具。"""

from distill.core.engine import DistillEngine
from distill.core.hooks import Hook, HookManager
from distill.core.registry import Registry
from distill.core.structures import InstancePrediction, PseudoLabelInstance

__all__ = [
    "DistillEngine",
    "Hook",
    "HookManager",
    "Registry",
    "InstancePrediction",
    "PseudoLabelInstance",
]
