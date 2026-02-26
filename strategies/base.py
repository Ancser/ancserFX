"""
Base strategy abstract class and parameter descriptor for the strategy layer.

All strategies inherit from BaseStrategy and declare their tunable parameters
as StrategyParam instances.  The parameter schema is used by the API layer
to render dynamic configuration UIs and by the backtest engine for
parameter sweeps.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from backtest.events import OrderEvent, SignalEvent


# ---------------------------------------------------------------------------
# Parameter descriptor
# ---------------------------------------------------------------------------


@dataclass
class StrategyParam:
    """Describes a single tunable strategy parameter.

    Attributes:
        name:        Human-readable label for the parameter.
        param_type:  One of "int", "float", "bool", "select".
        default:     Default value used when no override is supplied.
        min_val:     Lower bound (inclusive) for numeric types.
        max_val:     Upper bound (inclusive) for numeric types.
        step:        Step size for numeric sliders / spinboxes.
        options:     Allowed values when param_type is "select".
        description: Tooltip / help text displayed in the UI.
    """

    name: str
    param_type: str  # "int", "float", "bool", "select"
    default: Any
    min_val: Any = None
    max_val: Any = None
    step: Any = None
    options: list[Any] | None = None
    description: str = ""

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dictionary for API responses."""
        d: dict[str, Any] = {
            "name": self.name,
            "type": self.param_type,
            "default": self.default,
            "description": self.description,
        }
        if self.min_val is not None:
            d["min"] = self.min_val
        if self.max_val is not None:
            d["max"] = self.max_val
        if self.step is not None:
            d["step"] = self.step
        if self.options is not None:
            d["options"] = self.options
        return d

    def validate(self, value: Any) -> Any:
        """Coerce *value* to the declared type and enforce range constraints.

        Returns the coerced value or raises ``ValueError`` on failure.
        """
        coerced: Any
        if self.param_type == "int":
            coerced = int(value)
        elif self.param_type == "float":
            coerced = float(value)
        elif self.param_type == "bool":
            if isinstance(value, str):
                coerced = value.lower() in ("true", "1", "yes")
            else:
                coerced = bool(value)
        elif self.param_type == "select":
            if self.options is not None and value not in self.options:
                raise ValueError(
                    f"Parameter '{self.name}' must be one of {self.options}, "
                    f"got {value!r}"
                )
            coerced = value
        else:
            coerced = value

        # Range enforcement for numeric types
        if self.param_type in ("int", "float"):
            if self.min_val is not None and coerced < self.min_val:
                raise ValueError(
                    f"Parameter '{self.name}' must be >= {self.min_val}, "
                    f"got {coerced}"
                )
            if self.max_val is not None and coerced > self.max_val:
                raise ValueError(
                    f"Parameter '{self.name}' must be <= {self.max_val}, "
                    f"got {coerced}"
                )

        return coerced


# ---------------------------------------------------------------------------
# Abstract base strategy
# ---------------------------------------------------------------------------


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies.

    Subclasses **must** define the class-level attributes ``name``,
    ``description``, ``category``, and ``params``, and implement
    :meth:`on_bar`.

    The backtest engine calls :meth:`on_init` once before iteration and
    then :meth:`on_bar` for every incoming bar.
    """

    name: str = "Unnamed"
    description: str = ""
    category: str = "basic"  # "basic" or "orderflow"
    params: dict[str, StrategyParam] = {}

    def __init__(self, **kwargs: Any) -> None:
        """Instantiate a strategy, applying *kwargs* over declared defaults.

        Unknown keys are silently ignored so callers can pass a superset of
        parameters (e.g. from a saved JSON config).
        """
        self._param_values: dict[str, Any] = {}
        for key, param in self.params.items():
            raw = kwargs.get(key, param.default)
            self._param_values[key] = param.validate(raw)

    # -- helpers -------------------------------------------------------------

    def get_param(self, key: str) -> Any:
        """Return the runtime value of parameter *key*."""
        return self._param_values[key]

    # -- lifecycle hooks -----------------------------------------------------

    @abstractmethod
    def on_bar(self, bar: dict, history: pd.DataFrame) -> SignalEvent | None:
        """Process a single bar and optionally emit a signal.

        Parameters:
            bar:     Dict with keys ``timestamp``, ``open``, ``high``,
                     ``low``, ``close``, ``volume``.
            history: DataFrame of all bars seen so far (inclusive of *bar*).

        Returns:
            A :class:`SignalEvent` when the strategy identifies an
            opportunity, or ``None`` to stay flat / hold the current
            position.
        """
        ...

    def on_init(self, data: pd.DataFrame) -> None:
        """Called once before the backtest begins.

        Override to pre-compute indicators on the full dataset so that
        :meth:`on_bar` can look them up by index rather than recalculating
        every bar.
        """
        pass

    def build_order(
        self,
        signal: SignalEvent,
        bar: dict,
        quantity: int,
        tick_size: float,
    ) -> OrderEvent | None:
        """Optionally create a custom OrderEvent with SL/TP levels.

        Override when a strategy needs to attach stop-loss, take-profit,
        or multi-level take-profit orders.  The engine calls this after
        on_bar() returns a non-None signal.

        If this returns None (default), the engine uses Portfolio.on_signal().

        Args:
            signal:    The signal returned by on_bar().
            bar:       Current bar dict with OHLCV data.
            quantity:  Contract count from BacktestConfig.
            tick_size: Instrument tick size for price calculations.
        """
        return None

    def on_position_closed(self) -> None:
        """Called by the engine when the position goes flat after fills.

        Override to reset internal position-tracking state.
        """
        pass

    # -- introspection -------------------------------------------------------

    @classmethod
    def get_param_schema(cls) -> dict[str, dict]:
        """Return the parameter schema as a JSON-serializable dict."""
        return {k: v.to_dict() for k, v in cls.params.items()}

    @classmethod
    def get_info(cls) -> dict:
        """Return full strategy metadata including parameters."""
        return {
            "name": cls.name,
            "description": cls.description,
            "category": cls.category,
            "parameters": cls.get_param_schema(),
        }
