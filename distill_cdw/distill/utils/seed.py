"""可复现性相关的随机种子工具。"""

from __future__ import annotations

import os
import random
from typing import Optional

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy 可选
    np = None  # type: ignore

try:
    import torch
except Exception:  # pragma: no cover - torch 可选
    torch = None  # type: ignore


def set_seed(seed: int, deterministic: bool = False) -> None:
    """设置 Python、numpy 与 torch 的随机种子。"""

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
