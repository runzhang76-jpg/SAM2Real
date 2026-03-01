"""配置 IO 工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from distill.utils.logging import get_logger


def load_config(path: str) -> Dict[str, Any]:
    """从 YAML 或 JSON 加载配置。

    如果 PyYAML 可用，则使用 YAML 解析，否则尝试 JSON 解析。
    """

    logger = get_logger("distill")
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    text = file_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except Exception:
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse config. Install PyYAML or use JSON-compatible YAML.")
            raise exc


def save_config(cfg: Dict[str, Any], path: str) -> None:
    """将配置快照保存为 JSON。"""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
