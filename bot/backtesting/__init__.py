"""Backtesting: motor por eventos, metricas, walk-forward y Monte Carlo."""

from bot.backtesting.benchmark import BenchmarkComparison, buy_and_hold, compare
from bot.backtesting.engine import BacktestEngine, BacktestResult
from bot.backtesting.metrics import (
    Metrics,
    compute_metrics,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    trades_to_frame,
)
from bot.backtesting.montecarlo import MonteCarloResult, run_monte_carlo
from bot.backtesting.walkforward import (
    FoldResult,
    WalkForwardAnalysis,
    WalkForwardReport,
    Window,
    build_windows,
    expand_grid,
)

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "BenchmarkComparison",
    "FoldResult",
    "Metrics",
    "MonteCarloResult",
    "WalkForwardAnalysis",
    "WalkForwardReport",
    "Window",
    "build_windows",
    "buy_and_hold",
    "compare",
    "compute_metrics",
    "expand_grid",
    "max_drawdown",
    "run_monte_carlo",
    "sharpe_ratio",
    "sortino_ratio",
    "trades_to_frame",
]
