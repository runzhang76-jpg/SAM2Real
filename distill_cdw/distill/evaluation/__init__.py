"""蒸馏评估工具。"""

from distill.evaluation.coco_eval import CocoEvaluator
from distill.evaluation.metrics import compute_pseudolabel_stats

__all__ = [
    "CocoEvaluator",
    "compute_pseudolabel_stats",
]
