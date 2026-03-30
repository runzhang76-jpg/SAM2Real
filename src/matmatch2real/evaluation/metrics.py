""""""

from __future__ import annotations

from typing import Dict, List

from matmatch2real.core.structures import PseudoLabelInstance


def compute_pseudolabel_stats(batch_instances: List[List[PseudoLabelInstance]]) -> Dict[str, float]:
    """ batch """

    total_instances = sum(len(insts) for insts in batch_instances)
    if total_instances == 0:
        return {
            "pl_count": 0.0,
            "pl_avg_score": 0.0,
            "pl_avg_reliability": 0.0,
        }

    total_score = sum(inst.score for insts in batch_instances for inst in insts)
    total_rel = sum(inst.reliability for insts in batch_instances for inst in insts)
    return {
        "pl_count": float(total_instances),
        "pl_avg_score": float(total_score / total_instances),
        "pl_avg_reliability": float(total_rel / total_instances),
    }
