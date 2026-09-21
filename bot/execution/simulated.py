"""Contabilidad compartida por los brokers simulados (backtest y paper).

Backtest y paper difieren solo en de donde sale el precio y en el reloj: el
resto (efectivo, posiciones, comisiones, deslizamiento, gestion de stops) es
identico, y conviene que lo sea para que los resultados sean comparables.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Mapping

import pandas as pd

from bot.data.filters import SymbolFilters
from bot.execution.base import (
    AccountState,
    Broker,
    ClosedTrade,
    ExecutionError,
    Fill,
    Order,
    OrderStatus,
    OrderType,
    Position,
    Side,
)

log = logging.getLogger(__name__)


class SimulatedBroker(Broker):
    """Motor de cuenta simulada con comisiones y deslizamiento explicitos."""

    name = "simulated"

    def __init__(
        self,
        *,
        initial_balance: float,
        fee_bps: float = 7.5,
        slippage_bps: float = 4.0,
        config_hash: str = "",
        spot_only: bool = True,
    ) -> None:
        if initial_balance <= 0:
            raise ValueError("el saldo inicial debe ser positivo")
        self.initial_balance = float(initial_balance)
        self.cash = float(initial_balance)
        self.fee_rate = fee_bps / 10_000.0
        self.slippage_rate = slippage_bps / 10_000.0
        self.config_hash = config_hash
        self.spot_only = spot_only

        self._positions: Dict[str, Position] = {}
        self._filters: Dict[str, SymbolFilters] = {}
        self._prices: Dict[str, float] = {}
        self.closed_trades: List[ClosedTrade] = []
        self.fills: List[Fill] = []
        self.now: pd.Timestamp = pd.Timestamp.now(tz="UTC")

    # ------------------------------------------------------------------ estado

    def account(self) -> AccountState:
        return AccountState(cash=self.cash, equity=self.equity(), positions=tuple(self._positions.values()))

    def equity(self) -> float:
        """Efectivo mas el valor de mercado de las posiciones abiertas."""
        unrealized = sum(
            position.unrealized_pnl(self._prices.get(symbol, position.entry_price))
            for symbol, position in self._positions.items()
        )
        return self.cash + sum(p.notional for p in self._positions.values()) + unrealized

    def price(self, symbol: str) -> float:
        key = symbol.upper()
        if key not in self._prices:
            raise ExecutionError(f"sin precio conocido para {key}")
        return self._prices[key]

    def filters(self, symbol: str) -> SymbolFilters:
        return self._filters.get(symbol.upper(), SymbolFilters.permissive(symbol))

    def set_filters(self, symbol: str, filters: SymbolFilters) -> None:
        self._filters[symbol.upper()] = filters

    def set_price(self, symbol: str, price: float) -> None:
        self._prices[symbol.upper()] = float(price)

    # ------------------------------------------------------------------ ordenes

    def fill_price(self, symbol: str, side: Side, *, reference: float | None = None) -> float:
        """Precio efectivo tras el deslizamiento, siempre en nuestra contra."""
        base = reference if reference is not None else self.price(symbol)
        return base * (1.0 + self.slippage_rate * side.sign)

    def submit(self, order: Order) -> Fill:
        symbol = order.symbol.upper()
        reference = order.price if order.order_type is OrderType.LIMIT else None
        price = self.fill_price(symbol, order.side, reference=reference)
        notional = price * order.quantity
        fee = notional * self.fee_rate

        if not order.reduce_only and notional + fee > self.cash + 1e-9:
            raise ExecutionError(
                f"efectivo insuficiente para {symbol}: hacen falta {notional + fee:.2f}, hay {self.cash:.2f}"
            )

        fill = Fill(
            order_id=order.client_id,
            symbol=symbol,
            side=order.side,
            quantity=order.quantity,
            price=price,
            fee=fee,
            timestamp=self.now,
            status=OrderStatus.FILLED,
            raw={"simulated": True, "slippage_rate": self.slippage_rate},
        )
        self.fills.append(fill)
        return fill

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
        symbol = symbol.upper()
        if self.spot_only and side is Side.SHORT:
            # El simulador imita spot: sin margen no se puede vender lo que no
            # se tiene, y un backtest que lo permitiera daria resultados que el
            # broker real jamas podria reproducir.
            raise ExecutionError(
                f"{symbol}: cortos bloqueados por risk.spot_only; el spot no permite vender en descubierto"
            )
        if symbol in self._positions:
            raise ExecutionError(f"ya hay una posicion abierta en {symbol}")

        filters = self.filters(symbol)
        quantity = filters.round_qty(quantity)
        if quantity <= 0:
            raise ExecutionError(f"{symbol}: la cantidad se anula al aplicar el step del exchange")

        fill = self.submit(Order(symbol=symbol, side=side, quantity=quantity))
        self.cash -= fill.notional + fill.fee

        position = Position(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=fill.price,
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            opened_at=self.now,
            fees_paid=fill.fee,
            config_hash=self.config_hash,
            metadata=dict(metadata or {}),
        )
        self._positions[symbol] = position
        log.info(
            "ABIERTA %s %s qty=%.8f @ %.6f stop=%.6f tp=%.6f",
            side.value, symbol, quantity, fill.price, stop_loss, take_profit,
        )
        return position

    def close_position(self, symbol: str, *, reason: str = "manual", price: float | None = None) -> Fill | None:
        symbol = symbol.upper()
        position = self._positions.get(symbol)
        if position is None:
            return None

        exit_side = position.side.opposite
        fill_price = self.fill_price(symbol, exit_side, reference=price)
        notional = fill_price * position.quantity
        fee = notional * self.fee_rate

        fill = Fill(
            order_id=f"close-{symbol}-{len(self.fills)}",
            symbol=symbol,
            side=exit_side,
            quantity=position.quantity,
            price=fill_price,
            fee=fee,
            timestamp=self.now,
            raw={"reason": reason},
        )
        self.fills.append(fill)

        # El efectivo recupera el coste de entrada mas el resultado de la
        # operacion, menos la comision de salida.
        gross_pnl = (fill_price - position.entry_price) * position.quantity * position.side.sign
        self.cash += position.notional + gross_pnl - fee

        risk = position.risk_per_unit * position.quantity
        net_pnl = gross_pnl - fee - position.fees_paid
        trade = ClosedTrade(
            symbol=symbol,
            side=position.side,
            quantity=position.quantity,
            entry_price=position.entry_price,
            exit_price=fill_price,
            opened_at=position.opened_at,
            closed_at=self.now,
            pnl=net_pnl,
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
            fees=fee + position.fees_paid,
            r_multiple=net_pnl / risk if risk > 0 else 0.0,
            exit_reason=reason,
            config_hash=position.config_hash,
        )
        self.closed_trades.append(trade)
        del self._positions[symbol]

        log.info("CERRADA %s por %s @ %.6f pnl=%.2f (%.2fR)", symbol, reason, fill_price, net_pnl, trade.r_multiple)
        return fill

    # ------------------------------------------------------------------ gestion

    def check_exits(self, bars: Mapping[str, Mapping[str, float]]) -> List[ClosedTrade]:
        """Aplica stops y objetivos usando el rango de cada vela.

        Si una misma vela toca stop y objetivo se asume el stop: sin datos de
        tick no se puede saber el orden, y suponer lo peor evita que el
        backtest sea mas optimista que la realidad.
        """
        closed: List[ClosedTrade] = []
        for symbol in list(self._positions):
            bar = bars.get(symbol)
            if bar is None:
                continue
            position = self._positions[symbol]
            low, high = float(bar["low"]), float(bar["high"])

            if position.hit_stop(low, high):
                self.close_position(symbol, reason="stop_loss", price=position.stop_loss)
                closed.append(self.closed_trades[-1])
            elif position.hit_target(low, high):
                self.close_position(symbol, reason="take_profit", price=position.take_profit)
                closed.append(self.closed_trades[-1])
        return closed

    def close_all(self, reason: str = "session_end") -> List[ClosedTrade]:
        closed: List[ClosedTrade] = []
        for symbol in list(self._positions):
            self.close_position(symbol, reason=reason)
            closed.append(self.closed_trades[-1])
        return closed
