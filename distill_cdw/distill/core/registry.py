"""可插拔组件的简单注册表。"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type


class Registry:
    """最小化的 name -> callable 注册表。"""

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: Dict[str, Callable[..., Any]] = {}

    def register(self, key: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """用于注册可调用对象的装饰器。"""

        def _wrapper(obj: Callable[..., Any]) -> Callable[..., Any]:
            if key in self._items:
                raise KeyError(f"{self.name} registry already has key: {key}")
            self._items[key] = obj
            return obj

        return _wrapper

    def get(self, key: str) -> Callable[..., Any]:
        """获取已注册的可调用对象。"""
        if key not in self._items:
            raise KeyError(f"{self.name} registry missing key: {key}")
        return self._items[key]

    def build(self, key: str, *args: Any, **kwargs: Any) -> Any:
        """实例化已注册类或调用工厂函数。"""
        builder = self.get(key)
        return builder(*args, **kwargs)

    def list_keys(self) -> Dict[str, Callable[..., Any]]:
        """返回所有已注册项。"""
        return dict(self._items)
