"""Custom soft-distillation route."""

__all__ = [
    "CocoEvaluator",
    "DistillEngine",
    "DistillLoss",
    "EngineState",
    "build_eval_dataloader",
    "build_student",
    "build_train_dataloader",
    "prepare_class_mappings",
]


def __getattr__(name: str):
    if name in {"DistillEngine", "EngineState"}:
        from matmatch2real.core.engine import DistillEngine, EngineState

        return {"DistillEngine": DistillEngine, "EngineState": EngineState}[name]
    if name in {"build_eval_dataloader", "build_train_dataloader", "prepare_class_mappings"}:
        from matmatch2real.data.datasets import build_eval_dataloader, build_train_dataloader, prepare_class_mappings

        return {
            "build_eval_dataloader": build_eval_dataloader,
            "build_train_dataloader": build_train_dataloader,
            "prepare_class_mappings": prepare_class_mappings,
        }[name]
    if name == "CocoEvaluator":
        from matmatch2real.evaluation.coco_eval import CocoEvaluator

        return CocoEvaluator
    if name == "DistillLoss":
        from matmatch2real.student.losses import DistillLoss

        return DistillLoss
    if name == "build_student":
        from matmatch2real.student.models import build_student

        return build_student
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
