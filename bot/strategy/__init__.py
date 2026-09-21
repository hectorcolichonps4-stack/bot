"""Estrategia: filtros obligatorios + puntuacion -> senal candidata."""

from bot.strategy.engine import StrategyEngine
from bot.strategy.features import (
    IncompleteSnapshot,
    MarketSnapshot,
    SnapshotBuilder,
    build_features,
    snapshot,
)
from bot.strategy.gates import DEFAULT_GATES, GateResult, first_failure, run_gates
from bot.strategy.scoring import DEFAULT_COMPONENTS, ScoreBreakdown, Scorer
from bot.strategy.signals import Decision, Rejection, Signal

__all__ = [
    "DEFAULT_COMPONENTS",
    "DEFAULT_GATES",
    "Decision",
    "GateResult",
    "IncompleteSnapshot",
    "MarketSnapshot",
    "Rejection",
    "ScoreBreakdown",
    "Scorer",
    "Signal",
    "SnapshotBuilder",
    "StrategyEngine",
    "build_features",
    "first_failure",
    "run_gates",
    "snapshot",
]
