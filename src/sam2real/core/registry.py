""""""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type


class Registry:
    """ name -> callable """

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: Dict[str, Callable[..., Any]] = {}

    def register(self, key: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """"""

        def _wrapper(obj: Callable[..., Any]) -> Callable[..., Any]:
            if key in self._items:
                raise KeyError(f"{self.name} registry already has key: {key}")
            self._items[key] = obj
            return obj

        return _wrapper

    def get(self, key: str) -> Callable[..., Any]:
        """"""
        if key not in self._items:
            raise KeyError(f"{self.name} registry missing key: {key}")
        return self._items[key]

    def build(self, key: str, *args: Any, **kwargs: Any) -> Any:
        """"""
        builder = self.get(key)
        return builder(*args, **kwargs)

    def list_keys(self) -> Dict[str, Callable[..., Any]]:
        """"""
        return dict(self._items)
