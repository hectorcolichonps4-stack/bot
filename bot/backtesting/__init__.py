"""Backtesting: motor por eventos, metricas, walk-forward y Monte Carlo."""

from bot.backtesting.baselines import (
    BaselineResult,
    make_ema_rule,
    make_rsi_macd_rule,
    run_baseline,
    rule_buy_and_hold,
    standard_baselines,
)
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
from bot.backtesting.reporting import (
    comparison_table,
    equity_to_frame,
    monthly_activity_summary,
    split_by_period,
    trades_per_month,
)
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
    "BaselineResult",
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
    "comparison_table",
    "equity_to_frame",
    "make_ema_rule",
    "make_rsi_macd_rule",
    "monthly_activity_summary",
    "rule_buy_and_hold",
    "run_baseline",
    "split_by_period",
    "standard_baselines",
    "trades_per_month",
    "compute_metrics",
    "expand_grid",
    "max_drawdown",
    "run_monte_carlo",
    "sharpe_ratio",
    "sortino_ratio",
    "trades_to_frame",
]
