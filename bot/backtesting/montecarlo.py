"""Monte Carlo sobre las operaciones del backtest.

El orden concreto en el que salieron las operaciones es una casualidad. Al
remuestrearlas se obtiene el abanico de resultados que la misma estrategia
podria haber dado, que es mucho mas informativo que una unica curva.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from bot.risk.ledger import ClosedTrade


@dataclass(frozen=True)
class MonteCarloResult:
    """Distribucion de resultados sobre ``iterations`` reordenaciones."""

    iterations: int
    final_equity: np.ndarray
    max_drawdowns: np.ndarray
    initial_balance: float

    @property
    def median_final(self) -> float:
        return float(np.median(self.final_equity))

    @property
    def median_drawdown_pct(self) -> float:
        return float(np.median(self.max_drawdowns))

    def percentile_final(self, q: float) -> float:
        return float(np.percentile(self.final_equity, q))

    def percentile_drawdown(self, q: float) -> float:
        return float(np.percentile(self.max_drawdowns, q))

    @property
    def probability_of_loss(self) -> float:
        """Fraccion de escenarios que terminan por debajo del capital inicial."""
        if self.final_equity.size == 0:
            return 0.0
        return float(np.mean(self.final_equity < self.initial_balance))

    def risk_of_ruin(self, threshold_pct: float = 50.0) -> float:
        """Probabilidad de perder mas del ``threshold_pct`` del capital."""
        if self.final_equity.size == 0:
            return 0.0
        floor = self.initial_balance * (1.0 - threshold_pct / 100.0)
        return float(np.mean(self.final_equity <= floor))

    def summary(self) -> str:
        return (
            f"{self.iterations} simulaciones | mediana {self.median_final:,.0f} | "
            f"p5 {self.percentile_final(5):,.0f} | p95 {self.percentile_final(95):,.0f} | "
            f"maxDD mediano {self.median_drawdown_pct:.2f}% (p95 {self.percentile_drawdown(95):.2f}%) | "
            f"prob. de perder {self.probability_of_loss * 100:.1f}% | "
            f"riesgo de ruina(50%) {self.risk_of_ruin() * 100:.1f}%"
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"final_equity": self.final_equity, "max_drawdown_pct": self.max_drawdowns})


def run_monte_carlo(
    trades: Sequence[ClosedTrade],
    *,
    initial_balance: float,
    iterations: int = 1000,
    seed: int = 7,
    bootstrap: bool = True,
) -> MonteCarloResult:
    """Remuestrea las operaciones y mide capital final y drawdown de cada serie.

    Con ``bootstrap=True`` se muestrea con reemplazo (admite rachas que el
    historico no llego a mostrar); con ``False`` solo se baraja el orden real.
    """
    returns = np.array([t.pnl / initial_balance for t in trades], dtype="float64")
    if returns.size == 0 or iterations <= 0:
        return MonteCarloResult(0, np.array([]), np.array([]), initial_balance)

    rng = np.random.default_rng(seed)
    finals = np.empty(iterations, dtype="float64")
    drawdowns = np.empty(iterations, dtype="float64")

    for i in range(iterations):
        sample = rng.choice(returns, size=returns.size, replace=True) if bootstrap else rng.permutation(returns)
        # Se compone sobre el capital vivo: una perdida temprana reduce el
        # tamano de todo lo que viene despues, igual que en la realidad.
        equity = initial_balance * np.cumprod(1.0 + sample)
        equity = np.concatenate([[initial_balance], equity])
        peak = np.maximum.accumulate(equity)
        finals[i] = equity[-1]
        drawdowns[i] = float(np.max((peak - equity) / peak) * 100.0)

    return MonteCarloResult(iterations, finals, drawdowns, initial_balance)
