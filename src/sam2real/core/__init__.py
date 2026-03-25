"""Core primitives for the custom soft-distill route."""

from sam2real.core.engine import DistillEngine, EngineState
from sam2real.core.hooks import (
    CheckpointHook,
    EvalHook,
    Hook,
    HookManager,
    LoggingHook,
    ProgressBarHook,
    TensorboardHook,
    VisualizationHook,
)
from sam2real.core.registry import Registry
from sam2real.core.structures import InstancePrediction, PseudoLabelInstance

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
