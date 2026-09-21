"""Memoria de operaciones cerradas sobre la que deciden los limites."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd

from bot.execution.base import ClosedTrade

__all__ = ["ClosedTrade", "Ledger", "day_start", "week_start"]


@dataclass
class Ledger:
    """Historial en memoria de la sesion, con consultas por ventana temporal."""

    trades: List[ClosedTrade] = field(default_factory=list)
    equity_high_water: float = 0.0
    streak_floor: int = 0
    """Indice desde el que se cuenta la racha de perdidas (ver ``reset_streak``)."""

    def record(self, trade: ClosedTrade) -> None:
        self.trades.append(trade)

    def mark_equity(self, equity: float) -> None:
        self.equity_high_water = max(self.equity_high_water, equity)

    def drawdown_pct(self, equity: float) -> float:
        if self.equity_high_water <= 0:
            return 0.0
        return max(0.0, (self.equity_high_water - equity) / self.equity_high_water * 100.0)

    # --------------------------------------------------------------- ventanas

    def since(self, moment: pd.Timestamp) -> List[ClosedTrade]:
        return [t for t in self.trades if t.closed_at >= moment]

    def pnl_since(self, moment: pd.Timestamp) -> float:
        return sum(t.pnl for t in self.since(moment))

    def count_since(self, moment: pd.Timestamp) -> int:
        return len(self.since(moment))

    def consecutive_losses(self) -> int:
        """Racha de perdidas contando desde la ultima operacion hacia atras."""
        streak = 0
        for trade in reversed(self.trades[self.streak_floor:]):
            if trade.is_win:
                break
            streak += 1
        return streak

    def reset_streak(self) -> None:
        """Da por cerrada la racha actual; la usa el rearme del kill switch.

        Sin esto, rearmar el interruptor lo volveria a disparar en el acto:
        las perdidas que lo activaron siguen siendo las ultimas del historial.
        """
        self.streak_floor = len(self.trades)

    def last_trade_for(self, symbol: str) -> ClosedTrade | None:
        for trade in reversed(self.trades):
            if trade.symbol == symbol.upper():
                return trade
        return None

    def last_loss(self) -> ClosedTrade | None:
        for trade in reversed(self.trades):
            if not trade.is_win:
                return trade
        return None


def day_start(moment: pd.Timestamp) -> pd.Timestamp:
    """Inicio del dia UTC. El bot vive en UTC para no depender del huso local."""
    return pd.Timestamp(moment).tz_convert("UTC").normalize()


def week_start(moment: pd.Timestamp) -> pd.Timestamp:
    """Lunes 00:00 UTC de la semana en curso."""
    start = day_start(moment)
    return start - pd.Timedelta(days=int(start.dayofweek))
