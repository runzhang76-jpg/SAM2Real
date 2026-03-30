"""Core primitives for the custom soft-distill route."""

from matmatch2real.core.engine import DistillEngine, EngineState
from matmatch2real.core.hooks import (
    CheckpointHook,
    EvalHook,
    Hook,
    HookManager,
    LoggingHook,
    ProgressBarHook,
    TensorboardHook,
    VisualizationHook,
)
from matmatch2real.core.registry import Registry
from matmatch2real.core.structures import InstancePrediction, PseudoLabelInstance

__all__ = [
    "CheckpointHook",
    "DistillEngine",
    "EngineState",
    "EvalHook",
    "Hook",
    "HookManager",
    "InstancePrediction",
    "LoggingHook",
    "ProgressBarHook",
    "PseudoLabelInstance",
    "Registry",
    "TensorboardHook",
    "VisualizationHook",
]
