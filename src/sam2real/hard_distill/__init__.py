"""Official YOLO hard-label training route."""

from sam2real.hard_distill.manifests import build_pseudo_stage_yaml
from sam2real.hard_distill.shot_sampler import sample_k_shot_records
from sam2real.hard_distill.yolo_runner import main, run_yolo_training

__all__ = ["build_pseudo_stage_yaml", "main", "run_yolo_training", "sample_k_shot_records"]
