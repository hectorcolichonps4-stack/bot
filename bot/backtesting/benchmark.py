"""Comparacion contra comprar y mantener.

Una estrategia solo merece la pena si bate a la alternativa trivial despues de
comisiones. Este modulo calcula esa alternativa con el mismo capital inicial y
el mismo coste de entrada y salida.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from bot.backtesting.metrics import Metrics, compute_metrics


@dataclass(frozen=True)
class BenchmarkComparison:
    """Estrategia frente a benchmark, con las diferencias ya calculadas."""

    strategy: Metrics
    benchmark: Metrics
    symbol: str

    @property
    def excess_return_pct(self) -> float:
        return self.strategy.total_return_pct - self.benchmark.total_return_pct

    @property
    def drawdown_delta_pct(self) -> float:
        """Negativo es bueno: significa menos caida que el benchmark."""
        return self.strategy.max_drawdown_pct - self.benchmark.max_drawdown_pct

    @property
    def beats_benchmark(self) -> bool:
        return self.excess_return_pct > 0

    def summary(self) -> str:
        verdict = "por encima de" if self.beats_benchmark else "por debajo de"
        return (
            f"estrategia {self.strategy.total_return_pct:+.2f}% {verdict} "
            f"comprar y mantener {self.symbol} ({self.benchmark.total_return_pct:+.2f}%) | "
            f"exceso {self.excess_return_pct:+.2f}% | "
            f"maxDD {self.strategy.max_drawdown_pct:.2f}% vs {self.benchmark.max_drawdown_pct:.2f}%"
        )


def buy_and_hold(
    frame: pd.DataFrame,
    *,
    initial_balance: float,
    fee_bps: float = 7.5,
) -> pd.Series:
    """Curva de capital de comprar en la primera vela y vender en la ultima."""
    if frame.empty:
        return pd.Series(dtype="float64")

    closes = frame.set_index("open_time")["close"].astype("float64")
    fee_rate = fee_bps / 10_000.0
    invested = initial_balance * (1.0 - fee_rate)
    units = invested / float(closes.iloc[0])

    equity = units * closes
    # La comision de salida solo se paga al liquidar, en la ultima vela.
    equity.iloc[-1] = equity.iloc[-1] * (1.0 - fee_rate)
    return equity


def compare(
    strategy_equity: pd.Series,
    strategy_metrics: Metrics,
    benchmark_frame: pd.DataFrame,
    *,
    symbol: str,
    initial_balance: float,
    fee_bps: float = 7.5,
) -> BenchmarkComparison:
    """Alinea el benchmark al periodo de la estrategia y compara."""
    if strategy_equity.empty or benchmark_frame.empty:
        return BenchmarkComparison(strategy_metrics, Metrics(), symbol)

    start, end = strategy_equity.index[0], strategy_equity.index[-1]
    window = benchmark_frame[
        (benchmark_frame["open_time"] >= start) & (benchmark_frame["open_time"] <= end)
    ]
    equity = buy_and_hold(window, initial_balance=initial_balance, fee_bps=fee_bps)
    return BenchmarkComparison(strategy_metrics, compute_metrics(equity, []), symbol)
