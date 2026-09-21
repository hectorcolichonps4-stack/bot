"""Broker de paper trading: precios reales, dinero simulado.

Es identico al de backtest en su contabilidad, pero toma los precios del
proveedor en vivo y persiste cada operacion en SQLite, de modo que una sesion
de paper se puede analizar despues con las mismas herramientas que un
backtest.
"""

from __future__ import annotations

import logging
from typing import Mapping

import pandas as pd

from bot.data.feed import DataFeed
from bot.execution.base import Fill, Position, Side
from bot.execution.simulated import SimulatedBroker
from bot.logging.db import Database

log = logging.getLogger(__name__)


class PaperBroker(SimulatedBroker):
    """Simulacion en tiempo real conectada al feed de mercado y a la base."""

    name = "paper"

    def __init__(
        self,
        feed: DataFeed,
        *,
        initial_balance: float,
        fee_bps: float = 7.5,
        slippage_bps: float = 4.0,
        config_hash: str = "",
        db: Database | None = None,
        run_id: int | None = None,
        spot_only: bool = True,
    ) -> None:
        super().__init__(
            initial_balance=initial_balance,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            config_hash=config_hash,
            spot_only=spot_only,
        )
        self.feed = feed
        self.db = db
        self.run_id = run_id
        self._trade_ids: dict[str, int] = {}

    # ------------------------------------------------------------------ precios

    def refresh_prices(self, symbols: list[str] | None = None) -> None:
        """Toma el precio medio del libro para cada simbolo vigilado."""
        self.now = pd.Timestamp.now(tz="UTC")
        for symbol in symbols or self.feed.config.data.symbols:
            try:
                self.set_price(symbol, float(self.feed.ticker(symbol)["last"]))
            except Exception:  # noqa: BLE001 - un simbolo caido no para la sesion
                log.warning("no se pudo actualizar el precio de %s", symbol, exc_info=True)

    def price(self, symbol: str) -> float:
        key = symbol.upper()
        if key not in self._prices:
            self.set_price(key, float(self.feed.ticker(key)["last"]))
        return self._prices[key]

    def filters(self, symbol: str):
        return self.feed.filters(symbol)

    def sync(self) -> None:
        self.refresh_prices()

    # ---------------------------------------------------------------- registro

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
        position = super().open_position(
            symbol, side, quantity, stop_loss=stop_loss, take_profit=take_profit, metadata=metadata
        )
        if self.db is not None:
            self._trade_ids[position.symbol] = self.db.open_trade(
                symbol=position.symbol,
                side=position.side.value,
                quantity=position.quantity,
                entry_price=position.entry_price,
                stop_loss=position.stop_loss,
                take_profit=position.take_profit,
                opened_at=position.opened_at,
                mode=self.name,
                config_hash=self.config_hash,
                fees=position.fees_paid,
                signal_id=position.metadata.get("signal_id") if position.metadata else None,
                run_id=self.run_id,
            )
        return position

    def close_position(self, symbol: str, *, reason: str = "manual", price: float | None = None) -> Fill | None:
        fill = super().close_position(symbol, reason=reason, price=price)
        if fill is None:
            return None

        trade = self.closed_trades[-1]
        trade_id = self._trade_ids.pop(symbol.upper(), None)
        if self.db is not None and trade_id is not None:
            self.db.close_trade(
                trade_id,
                exit_price=trade.exit_price,
                closed_at=trade.closed_at,
                pnl=trade.pnl,
                fees=trade.fees,
                r_multiple=trade.r_multiple,
                exit_reason=trade.exit_reason,
            )
        return fill

    def check_live_exits(self) -> list:
        """Evalua stops y objetivos contra el ultimo precio conocido.

        En vivo no hay "rango de la vela": se usa el ultimo precio como high y
        low a la vez, que es la aproximacion honesta con datos de sondeo.
        """
        bars = {symbol: {"high": price, "low": price} for symbol, price in self._prices.items()}
        return self.check_exits(bars)
