"""Plugin registries for data sources and processing steps.

Usage
-----
# Register a source:
    from woodpecker.core.registry import SourceRegistry
    @SourceRegistry.register("my_source")
    class MySource(DataSource): ...

# Retrieve a source class:
    cls = SourceRegistry.get("my_source")
    source = cls()
"""

from __future__ import annotations

from typing import Callable, Dict, Type, TypeVar

T = TypeVar("T")


class _Registry:
    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._store: Dict[str, type] = {}

    def register(self, name: str) -> Callable[[Type[T]], Type[T]]:
        def decorator(cls: Type[T]) -> Type[T]:
            if name in self._store:
                raise ValueError(f"{self._kind} '{name}' already registered.")
            self._store[name] = cls
            return cls
        return decorator

    def get(self, name: str) -> type:
        if name not in self._store:
            available = list(self._store)
            raise KeyError(
                f"{self._kind} '{name}' not found. Available: {available}"
            )
        return self._store[name]

    def names(self) -> list:
        return list(self._store)


SourceRegistry = _Registry("DataSource")
StepRegistry = _Registry("ProcessingStep")
