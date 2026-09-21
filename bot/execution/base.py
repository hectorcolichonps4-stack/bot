"""Primitivas de ejecucion y la interfaz ``Broker``.

Aqui viven los tipos basicos que comparten estrategia, riesgo y ejecucion:
lado de la operacion, ordenes, fills y posiciones. Las tres implementaciones
(backtest, paper y live) se limitan a cumplir este contrato, de modo que el
resto del bot no sabe en que modo se esta ejecutando.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence

import pandas as pd

from bot.data.filters import SymbolFilters


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"

    @property
    def sign(self) -> int:
        """+1 para largo, -1 para corto: simplifica el calculo de PnL."""
        return 1 if self is Side.LONG else -1

    @property
    def opposite(self) -> "Side":
        return Side.SHORT if self is Side.LONG else Side.LONG


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    NEW = "new"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


class ExecutionError(RuntimeError):
    """El broker no pudo completar la operacion."""


@dataclass(frozen=True)
class Order:
    """Intencion de operar, antes de conocer su resultado."""

    symbol: str
    side: Side
    quantity: float
    order_type: OrderType = OrderType.MARKET
    price: float | None = None
    reduce_only: bool = False
    client_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"cantidad invalida en la orden: {self.quantity}")
        if self.order_type is OrderType.LIMIT and not self.price:
            raise ValueError("una orden limit necesita precio")


@dataclass(frozen=True)
class Fill:
    """Ejecucion real de una orden, con su coste explicito."""

    order_id: str
    symbol: str
    side: Side
    quantity: float
    price: float
    fee: float
    timestamp: pd.Timestamp
    status: OrderStatus = OrderStatus.FILLED
    raw: Mapping[str, object] = field(default_factory=dict)

    @property
    def notional(self) -> float:
        return self.price * self.quantity


@dataclass
class Position:
    """Posicion abierta con su gestion de salida asociada."""

    symbol: str
    side: Side
    quantity: float
    entry_price: float
    stop_loss: float
    take_profit: float
    opened_at: pd.Timestamp
    fees_paid: float = 0.0
    signal_id: int | None = None
    config_hash: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def notional(self) -> float:
        return self.entry_price * self.quantity

    @property
    def risk_per_unit(self) -> float:
        """Distancia al stop: la unidad "R" con la que se mide el resultado."""
        return abs(self.entry_price - self.stop_loss)

    def unrealized_pnl(self, price: float) -> float:
        return (price - self.entry_price) * self.quantity * self.side.sign

    def r_multiple(self, price: float) -> float:
        risk = self.risk_per_unit * self.quantity
        return self.unrealized_pnl(price) / risk if risk > 0 else 0.0

    def hit_stop(self, low: float, high: float) -> bool:
        return low <= self.stop_loss if self.side is Side.LONG else high >= self.stop_loss

    def hit_target(self, low: float, high: float) -> bool:
        return high >= self.take_profit if self.side is Side.LONG else low <= self.take_profit


@dataclass(frozen=True)
class AccountState:
    """Foto de la cuenta que consume el modulo de riesgo."""

    cash: float
    equity: float
    positions: Sequence[Position]

    @property
    def exposure(self) -> float:
        return sum(position.notional for position in self.positions)

    @property
    def exposure_pct(self) -> float:
        return (self.exposure / self.equity * 100.0) if self.equity > 0 else 0.0


@dataclass(frozen=True)
class ClosedTrade:
    """Operacion completa ya cerrada, con el PnL neto de comisiones.

    Vive junto a ``Fill`` y ``Position`` y no en ``/risk`` porque es el
    resultado que produce la ejecucion; el modulo de riesgo se limita a
    consumirlo para llevar sus limites.
    """

    symbol: str
    side: Side
    quantity: float
    entry_price: float
    exit_price: float
    opened_at: pd.Timestamp
    closed_at: pd.Timestamp
    pnl: float
    stop_loss: float = 0.0
    take_profit: float = 0.0
    fees: float = 0.0
    r_multiple: float = 0.0
    exit_reason: str = ""
    config_hash: str = ""

    @property
    def is_win(self) -> bool:
        return self.pnl > 0

    @property
    def duration(self) -> pd.Timedelta:
        return self.closed_at - self.opened_at


class Broker(ABC):
    """Contrato unico para backtest, paper y live.

    Toda la logica de negocio del bot se escribe contra esta interfaz; cambiar
    de modo es cambiar la implementacion inyectada, nada mas.
    """

    name: str = "broker"

    @abstractmethod
    def account(self) -> AccountState:
        """Efectivo, equity y posiciones abiertas."""

    @abstractmethod
    def price(self, symbol: str) -> float:
        """Ultimo precio negociable del simbolo."""

    @abstractmethod
    def filters(self, symbol: str) -> SymbolFilters:
        """Filtros del simbolo para redondear precio y cantidad."""

    @abstractmethod
    def submit(self, order: Order) -> Fill:
        """Envia la orden y devuelve su ejecucion."""

    @abstractmethod
    def open_position(
        self,
        symbol: str,
        side: Side,
        quantity: float,
        *,
        stop_loss: float,
        take_profit: float,
        metadata: Mapping[str, object] | None = None,
    ) -> Position:
        """Abre posicion y registra su gestion de salida."""

    @abstractmethod
    def close_position(self, symbol: str, *, reason: str = "manual") -> Fill | None:
        """Cierra la posicion del simbolo, si existe."""

    def positions(self) -> Sequence[Position]:
        return self.account().positions

    def position_for(self, symbol: str) -> Position | None:
        for position in self.positions():
            if position.symbol == symbol.upper():
                return position
        return None

    def sync(self) -> None:
        """Gancho opcional para refrescar el estado contra el exchange."""
        return None
