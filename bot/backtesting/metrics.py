"""Metricas de rendimiento calculadas sobre la curva de capital y los trades."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from bot.risk.ledger import ClosedTrade

PERIODS_PER_YEAR = 365.0

# Anualizar un tramo corto produce cifras que no significan nada: una corrida
# detenida a los 16 dias daria un "CAGR" de dos digitos que el lector
# compararia con el de cinco anos. Por debajo de este umbral no se anualiza.
MIN_DAYS_TO_ANNUALIZE = 180.0

# Por debajo de esto la dispersion es ruido de coma flotante, no volatilidad:
# la desviacion de una serie constante sale ~1e-19, no 0, y dividir por ella
# produce ratios astronomicos sin ningun significado.
_FLAT = 1e-12


@dataclass(frozen=True)
class Metrics:
    """Resumen de una ejecucion. Todos los porcentajes van en tanto por ciento."""

    initial_equity: float = 0.0
    final_equity: float = 0.0
    total_return_pct: float = 0.0
    cagr_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    expectancy_r: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    max_consecutive_losses: int = 0
    avg_duration_hours: float = 0.0
    total_fees: float = 0.0
    days: float = 0.0

    @property
    def annualizable(self) -> bool:
        """Si el periodo da para extrapolar a un ano sin decir tonterias."""
        return self.days >= MIN_DAYS_TO_ANNUALIZE

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def summary(self) -> str:
        cagr = f"CAGR {self.cagr_pct:+.2f}%" if self.annualizable else f"CAGR n/d ({self.days:.0f} dias)"
        return (
            f"retorno {self.total_return_pct:+.2f}% | {cagr} | "
            f"maxDD {self.max_drawdown_pct:.2f}% | Sharpe {self.sharpe:.2f} | "
            f"{self.trades} trades | acierto {self.win_rate_pct:.1f}% | "
            f"PF {self.profit_factor:.2f} | esperanza {self.expectancy_r:+.2f}R"
        )


def max_drawdown(equity: pd.Series) -> float:
    """Caida maxima desde maximo, en porcentaje."""
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return float(abs(drawdown.min()) * 100.0)


def sharpe_ratio(returns: pd.Series, *, periods_per_year: float = PERIODS_PER_YEAR, risk_free: float = 0.0) -> float:
    """Sharpe anualizado. Devuelve 0 si no hay dispersion que medir."""
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free / periods_per_year
    std = float(excess.std(ddof=1))
    if np.isnan(std) or std < _FLAT:
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, *, periods_per_year: float = PERIODS_PER_YEAR) -> float:
    """Como el Sharpe pero penalizando solo la volatilidad a la baja."""
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < 0]
    if downside.empty:
        return 0.0
    std = float(downside.std(ddof=1))
    if np.isnan(std) or std < _FLAT:
        return 0.0
    return float(returns.mean() / std * np.sqrt(periods_per_year))


def max_consecutive_losses(trades: Sequence[ClosedTrade]) -> int:
    worst = current = 0
    for trade in trades:
        current = 0 if trade.is_win else current + 1
        worst = max(worst, current)
    return worst


def compute_metrics(
    equity_curve: pd.Series,
    trades: Sequence[ClosedTrade],
    *,
    periods_per_year: float = PERIODS_PER_YEAR,
) -> Metrics:
    """Cruza curva de capital y operaciones en un unico resumen.

    ``equity_curve`` debe tener indice temporal. Si esta vacia se devuelven
    metricas neutras en vez de lanzar: un backtest sin operaciones es un
    resultado valido, no un error.
    """
    if equity_curve.empty:
        return Metrics(trades=len(trades))

    equity = equity_curve.astype("float64").dropna()
    initial, final = float(equity.iloc[0]), float(equity.iloc[-1])
    total_return = (final / initial - 1.0) * 100.0 if initial > 0 else 0.0

    index = pd.to_datetime(equity.index)
    days = max((index[-1] - index[0]).total_seconds() / 86400.0, 1e-9)
    years = days / 365.0
    cagr = ((final / initial) ** (1.0 / years) - 1.0) * 100.0 if initial > 0 and years > 0 else 0.0

    # Se remuestrea a diario para que el Sharpe no dependa del timeframe.
    daily = equity.resample("1D").last().dropna() if isinstance(equity.index, pd.DatetimeIndex) else equity
    returns = daily.pct_change().dropna()

    drawdown = max_drawdown(equity)
    wins = [t for t in trades if t.is_win]
    losses = [t for t in trades if not t.is_win]
    gross_profit = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))

    return Metrics(
        initial_equity=initial,
        final_equity=final,
        total_return_pct=total_return,
        cagr_pct=cagr,
        max_drawdown_pct=drawdown,
        sharpe=sharpe_ratio(returns, periods_per_year=periods_per_year),
        sortino=sortino_ratio(returns, periods_per_year=periods_per_year),
        calmar=(cagr / drawdown) if drawdown > 0 else 0.0,
        trades=len(trades),
        wins=len(wins),
        losses=len(losses),
        win_rate_pct=(len(wins) / len(trades) * 100.0) if trades else 0.0,
        profit_factor=(gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0),
        expectancy=(sum(t.pnl for t in trades) / len(trades)) if trades else 0.0,
        expectancy_r=(sum(t.r_multiple for t in trades) / len(trades)) if trades else 0.0,
        avg_win=(gross_profit / len(wins)) if wins else 0.0,
        avg_loss=(-gross_loss / len(losses)) if losses else 0.0,
        largest_win=max((t.pnl for t in wins), default=0.0),
        largest_loss=min((t.pnl for t in losses), default=0.0),
        max_consecutive_losses=max_consecutive_losses(trades),
        avg_duration_hours=(
            sum(t.duration.total_seconds() for t in trades) / len(trades) / 3600.0 if trades else 0.0
        ),
        total_fees=sum(t.fees for t in trades),
        days=days,
    )


def trades_to_frame(trades: Sequence[ClosedTrade]) -> pd.DataFrame:
    """Operaciones cerradas como DataFrame, para informes y dashboard."""
    if not trades:
        return pd.DataFrame(
            columns=[
                "symbol", "side", "quantity", "entry_price", "exit_price",
                "opened_at", "closed_at", "pnl", "fees", "r_multiple", "exit_reason",
            ]
        )
    return pd.DataFrame(
        [
            {
                "symbol": t.symbol,
                "side": t.side.value,
                "quantity": t.quantity,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "opened_at": t.opened_at,
                "closed_at": t.closed_at,
                "pnl": t.pnl,
                "fees": t.fees,
                "r_multiple": t.r_multiple,
                "exit_reason": t.exit_reason,
            }
            for t in trades
        ]
    )
