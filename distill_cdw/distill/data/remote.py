"""远程数据集同步（SwanLab 适配器）。"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from distill.utils.logging import get_logger


@dataclass
class RemoteConfig:
    """远程数据集同步的配置子集。"""

    enabled: bool
    provider: str
    remote_uri: str
    local_cache_dir: str
    sync_mode: str
    version: str = ""


class SwanLabUnavailable(RuntimeError):
    """当 SwanLab 未安装或未配置时抛出。"""


class SwanLabRemote:
    """SwanLab 远程数据集辅助类。"""

    def __init__(self) -> None:
        self.logger = get_logger("distill")
        self._cli_path = shutil.which("swanlab")
        self._sdk = self._try_import_swanlab()

    def _try_import_swanlab(self) -> Optional[Any]:
        try:
            import swanlab  # type: ignore

            return swanlab
        except Exception:
            return None

    @property
    def available(self) -> bool:
        return self._cli_path is not None or self._sdk is not None

    def _try_sdk_sync(
        self,
        mode: str,
        local_path: Path,
        remote_uri: str,
        includes: Optional[Iterable[str]] = None,
        excludes: Optional[Iterable[str]] = None,
        overwrite: bool = False,
    ) -> bool:
        if self._sdk is None:
            return False

        def _call(obj: Any, method: str) -> bool:
            func = getattr(obj, method, None)
            if not callable(func):
                return False
            try:
                func(
                    remote_uri=remote_uri,
                    local_path=str(local_path),
                    includes=includes,
                    excludes=excludes,
                    overwrite=overwrite,
                    mode=mode,
                )
                return True
            except Exception as exc:
                self.logger.warning("SwanLab SDK 调用失败(%s.%s): %s", obj, method, exc)
                return False

        for attr in ("dataset", "datasets"):
            obj = getattr(self._sdk, attr, None)
            if obj is None:
                continue
            if _call(obj, "sync"):
                return True
            if mode == "down" and _call(obj, "download"):
                return True
            if mode == "up" and _call(obj, "upload"):
                return True

        dataset_cls = getattr(self._sdk, "Dataset", None)
        if dataset_cls is not None:
            try:
                dataset = dataset_cls(remote_uri)
                if mode == "down" and _call(dataset, "download"):
                    return True
                if mode == "up" and _call(dataset, "upload"):
                    return True
            except Exception as exc:
                self.logger.warning("SwanLab SDK Dataset 初始化失败: %s", exc)

        return False

    def sync_up(
        self, local_path: Path, remote_uri: str, includes: Optional[Iterable[str]] = None, excludes: Optional[Iterable[str]] = None
    ) -> None:
        if not self.available:
            raise SwanLabUnavailable("SwanLab CLI/SDK not available")
        if self._try_sdk_sync("up", local_path, remote_uri, includes, excludes):
            self.logger.info("[swanlab] SDK 上传完成: %s -> %s", local_path, remote_uri)
            return
        if self._cli_path:
            self._run_cli("up", local_path, remote_uri, includes, excludes)
            return
        self.logger.warning("SwanLab CLI 不可用，无法上传数据集")

    def sync_down(self, remote_uri: str, local_path: Path, overwrite: bool = False) -> None:
        if not self.available:
            raise SwanLabUnavailable("SwanLab CLI/SDK not available")
        if self._try_sdk_sync("down", local_path, remote_uri, None, None, overwrite=overwrite):
            self.logger.info("[swanlab] SDK 下载完成: %s -> %s", remote_uri, local_path)
            return
        if self._cli_path:
            self._run_cli("down", local_path, remote_uri, None, None, overwrite=overwrite)
            return
        self.logger.warning("SwanLab CLI 不可用，无法下载数据集")

    def _run_cli(
        self,
        mode: str,
        local_path: Path,
        remote_uri: str,
        includes: Optional[Iterable[str]],
        excludes: Optional[Iterable[str]],
        overwrite: bool = False,
    ) -> None:
        # 注意：SwanLab 数据集 CLI 语法未确认前使用占位参数。
        cmd = [self._cli_path or "swanlab", "dataset", "sync", mode, "--uri", remote_uri, "--local", str(local_path)]
        if overwrite:
            cmd.append("--overwrite")
        for pattern in includes or []:
            cmd.extend(["--include", pattern])
        for pattern in excludes or []:
            cmd.extend(["--exclude", pattern])
        self.logger.debug("running SwanLab CLI: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except Exception as exc:
            self.logger.warning("SwanLab CLI failed: %s", exc)


def _scan_dir_stats(root: Path) -> Dict[str, Any]:
    file_count = 0
    total_size = 0
    for path in root.rglob("*"):
        if path.is_file():
            file_count += 1
            total_size += path.stat().st_size
    return {"file_count": file_count, "total_size_bytes": total_size}


def ensure_dataset_available(cfg: Dict[str, Any], output_dir: Optional[Path] = None) -> Path:
    """确保数据集在本地可用，并按需进行远程同步。

    返回解析后的本地根目录。
    """

    logger = get_logger("distill")
    data_cfg = cfg.get("data", {})
    remote_cfg = data_cfg.get("remote", {})
    train_root = data_cfg.get("train", {}).get("root_dir", data_cfg.get("root_dir", "."))
    rcfg = RemoteConfig(
        enabled=bool(remote_cfg.get("enabled", False)),
        provider=str(remote_cfg.get("provider", "")),
        remote_uri=str(remote_cfg.get("remote_uri", "")),
        local_cache_dir=str(remote_cfg.get("local_cache_dir", "")),
        sync_mode=str(remote_cfg.get("sync_mode", "none")),
        version=str(remote_cfg.get("version", "")),
    )

    if not rcfg.enabled:
        return Path(train_root)

    if rcfg.provider.lower() != "swanlab":
        logger.warning("remote provider not supported: %s", rcfg.provider)
        return Path(rcfg.local_cache_dir or train_root)

    if not rcfg.remote_uri:
        logger.warning("remote enabled but remote_uri is empty; using local paths")
        return Path(rcfg.local_cache_dir or train_root)

    local_cache = Path(rcfg.local_cache_dir)
    local_cache.mkdir(parents=True, exist_ok=True)

    swanlab = SwanLabRemote()
    if not swanlab.available:
        logger.warning(
            "SwanLab not available. Install with `pip install swanlab` or disable data.remote.enabled."
        )
        stats = _scan_dir_stats(local_cache) if local_cache.exists() else {"file_count": 0, "total_size_bytes": 0}
        manifest = {
            "provider": rcfg.provider,
            "remote_uri": rcfg.remote_uri,
            "version": rcfg.version,
            "local_cache_dir": str(local_cache),
            "status": "unavailable",
            "synced_at": datetime.utcnow().isoformat() + "Z",
            **stats,
        }
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = output_dir / "dataset_manifest.json"
            with manifest_path.open("w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        return local_cache

    if rcfg.sync_mode == "down":
        swanlab.sync_down(rcfg.remote_uri, local_cache, overwrite=False)
    elif rcfg.sync_mode == "up":
        swanlab.sync_up(local_cache, rcfg.remote_uri, includes=None, excludes=None)
    else:
        logger.info("remote sync_mode is 'none'; skip sync")

    stats = _scan_dir_stats(local_cache)
    manifest = {
        "provider": rcfg.provider,
        "remote_uri": rcfg.remote_uri,
        "version": rcfg.version,
        "local_cache_dir": str(local_cache),
        "synced_at": datetime.utcnow().isoformat() + "Z",
        **stats,
    }

    logger.info(
        "dataset remote_uri=%s version=%s cache=%s files=%d size=%d",
        rcfg.remote_uri,
        rcfg.version,
        local_cache,
        stats["file_count"],
        stats["total_size_bytes"],
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "dataset_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    return local_cache
