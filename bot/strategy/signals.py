"""Tipos que produce la estrategia: senal candidata o rechazo motivado."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import pandas as pd

from bot.execution.base import Side


@dataclass(frozen=True)
class Signal:
    """Senal candidata, ya puntuada pero todavia sin validar por riesgo."""

    symbol: str
    side: Side
    timestamp: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit: float
    score: float
    components: Mapping[str, float]
    snapshot: Mapping[str, object]
    config_hash: str = ""

    def __post_init__(self) -> None:
        if self.entry_price <= 0:
            raise ValueError("entry_price debe ser positivo")
        if self.side is Side.LONG and not (self.stop_loss < self.entry_price < self.take_profit):
            raise ValueError("en largo se requiere stop < entrada < objetivo")
        if self.side is Side.SHORT and not (self.take_profit < self.entry_price < self.stop_loss):
            raise ValueError("en corto se requiere objetivo < entrada < stop")

    @property
    def risk_per_unit(self) -> float:
        return abs(self.entry_price - self.stop_loss)

    @property
    def reward_risk(self) -> float:
        risk = self.risk_per_unit
        return abs(self.take_profit - self.entry_price) / risk if risk > 0 else 0.0


@dataclass(frozen=True)
class Rejection:
    """Motivo por el que no hubo senal. Se guarda siempre en la base de datos.

    El campo ``stage`` distingue quien rechazo (``gate``, ``score``, ``data``),
    porque un bot que no opera casi nunca falla por una sola razon.
    """

    symbol: str
    timestamp: pd.Timestamp
    stage: str
    reason: str
    detail: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    """Resultado de evaluar un simbolo: o senal, o rechazo, nunca ambos."""

    symbol: str
    timestamp: pd.Timestamp
    signal: Signal | None = None
    rejection: Rejection | None = None

    @property
    def accepted(self) -> bool:
        return self.signal is not None

    @classmethod
    def accept(cls, signal: Signal) -> "Decision":
        return cls(symbol=signal.symbol, timestamp=signal.timestamp, signal=signal)

    @classmethod
    def reject(cls, rejection: Rejection) -> "Decision":
        return cls(symbol=rejection.symbol, timestamp=rejection.timestamp, rejection=rejection)
