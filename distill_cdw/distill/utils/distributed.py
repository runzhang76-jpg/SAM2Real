"""分布式训练辅助（占位）。"""

from __future__ import annotations

import os
from typing import Any

try:
    import torch
    import torch.distributed as dist
except Exception:  # pragma: no cover - 可选依赖
    torch = None  # type: ignore
    dist = None  # type: ignore


def is_distributed() -> bool:
    return dist is not None and dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if not is_distributed():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed(backend: str = "nccl") -> None:
    """按环境变量初始化分布式进程组。"""

    if dist is None or torch is None:
        return
    if dist.is_initialized():
        return

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size <= 1:
        return

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    dist.init_process_group(backend=backend, init_method="env://", world_size=world_size, rank=rank)


def all_reduce(value: Any, op: str = "mean") -> Any:
    """对标量或张量进行跨进程归约。"""

    if not is_distributed() or dist is None or torch is None:
        return value

    is_tensor = isinstance(value, torch.Tensor)
    tensor = value if is_tensor else torch.tensor(value, device="cuda" if torch.cuda.is_available() else "cpu")

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if op == "mean":
        tensor = tensor / dist.get_world_size()

    if is_tensor:
        return tensor
    return tensor.item()
