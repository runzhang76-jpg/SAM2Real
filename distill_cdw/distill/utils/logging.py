"""日志工具。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

_LOGGER_CACHE = {}


def setup_logger(name: str, log_file: Optional[str] = None, level: str = "INFO") -> logging.Logger:
    """配置日志器，包含 stdout 与可选文件输出。"""

    if name in _LOGGER_CACHE:
        return _LOGGER_CACHE[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _LOGGER_CACHE[name] = logger
    return logger


def get_logger(name: str) -> logging.Logger:
    """获取日志器，必要时创建默认实例。"""

    if name in _LOGGER_CACHE:
        return _LOGGER_CACHE[name]
    return setup_logger(name)
