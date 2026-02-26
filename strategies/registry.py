"""
Strategy registry with auto-discovery.

On first access the registry scans ``strategies.basic`` and
``strategies.orderflow`` for concrete :class:`BaseStrategy` subclasses
and caches them by their ``name`` attribute.  The registry is the single
source of truth used by the API and backtest engine to instantiate
strategies by name.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import TYPE_CHECKING

from strategies.base import BaseStrategy

if TYPE_CHECKING:
    pass

_SCAN_PACKAGES = [
    "strategies.basic",
    "strategies.orderflow",
]


class StrategyRegistry:
    """Singleton-style registry of all available strategies."""

    _strategies: dict[str, type[BaseStrategy]] = {}
    _loaded: bool = False

    @classmethod
    def _discover(cls) -> None:
        """Lazily scan strategy packages and populate the registry."""
        if cls._loaded:
            return

        for package_name in _SCAN_PACKAGES:
            try:
                package = importlib.import_module(package_name)
            except ImportError:
                # Package may not exist yet during development
                continue

            if not hasattr(package, "__path__"):
                continue

            for _importer, modname, _ispkg in pkgutil.iter_modules(package.__path__):
                try:
                    module = importlib.import_module(f"{package_name}.{modname}")
                except ImportError:
                    continue

                for _attr_name, obj in inspect.getmembers(module, inspect.isclass):
                    if (
                        issubclass(obj, BaseStrategy)
                        and obj is not BaseStrategy
                        and hasattr(obj, "name")
                        and obj.name != "Unnamed"
                    ):
                        cls._strategies[obj.name] = obj

        cls._loaded = True

    # -- public API ----------------------------------------------------------

    @classmethod
    def list_strategies(cls, category: str | None = None) -> list[dict]:
        """Return metadata dicts for all (or category-filtered) strategies."""
        cls._discover()
        results: list[dict] = []
        for _name, strat_cls in cls._strategies.items():
            if category is not None and strat_cls.category != category:
                continue
            results.append(strat_cls.get_info())
        return results

    @classmethod
    def get_strategy(cls, name: str) -> type[BaseStrategy]:
        """Look up a strategy class by its ``name`` attribute.

        Raises:
            ValueError: If the strategy is not registered.
        """
        cls._discover()
        if name not in cls._strategies:
            available = sorted(cls._strategies.keys())
            raise ValueError(
                f"Strategy '{name}' not found. "
                f"Available: {available}"
            )
        return cls._strategies[name]

    @classmethod
    def reset(cls) -> None:
        """Clear the registry (useful for testing)."""
        cls._strategies.clear()
        cls._loaded = False
