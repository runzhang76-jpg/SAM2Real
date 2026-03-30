""""""

from __future__ import annotations

import os
import random
from typing import Optional

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy
    np = None  # type: ignore

try:
    import torch
except Exception:  # pragma: no cover - torch
    torch = None  # type: ignore


def set_seed(seed: int, deterministic: bool = False) -> None:
    """ Pythonnumpy  torch """

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
