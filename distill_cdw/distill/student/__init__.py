"""学生模型与损失定义。"""

from distill.student.base import StudentModel
from distill.student.models import build_student
from distill.student.losses import DistillLoss

__all__ = [
    "build_student",
    "StudentModel",
    "DistillLoss",
]
