""""""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

_LOGGER_CACHE = {}


def setup_logger(name: str, log_file: Optional[str] = None, level: str = "INFO") -> logging.Logger:
    """ stdout """

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
    """"""

    if name in _LOGGER_CACHE:
        return _LOGGER_CACHE[name]
    return setup_logger(name)
