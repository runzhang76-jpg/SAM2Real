"""教师端组件（SAM2、后处理、分类器）。"""

from distill.teacher.sam2_teacher import SAM2Teacher, build_teacher

__all__ = [
    "SAM2Teacher",
    "build_teacher",
]
