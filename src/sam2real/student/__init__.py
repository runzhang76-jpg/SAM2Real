"""Student-side building blocks for the custom distill route."""

__all__ = ["DistillLoss", "YOLOv8SegStudent", "build_student"]


def __getattr__(name: str):
    if name == "DistillLoss":
        from sam2real.student.losses import DistillLoss

        return DistillLoss
    if name == "YOLOv8SegStudent":
        from sam2real.student.yolov8_adapter import YOLOv8SegStudent

        return YOLOv8SegStudent
    if name == "build_student":
        from sam2real.student.models import build_student

        return build_student
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
