"""Gestion de riesgo: tamano, limites, correlacion, cooldowns y kill switch."""

from bot.risk.cooldown import CooldownCheck, check_global_cooldown, check_symbol_cooldown
from bot.risk.correlation import (
    CorrelationCheck,
    check_correlation,
    correlation_matrix,
    pairwise_correlation,
    returns_from_closes,
)
from bot.risk.kill_switch import KillSwitch, KillSwitchTrip
from bot.risk.ledger import ClosedTrade, Ledger, day_start, week_start
from bot.risk.limits import (
    LimitCheck,
    check_concurrent_positions,
    check_daily_loss,
    check_daily_trades,
    check_exposure,
    check_weekly_loss,
)
from bot.risk.manager import RiskDecision, RiskManager
from bot.risk.sizing import SizingResult, position_size

__all__ = [
    "ClosedTrade",
    "CooldownCheck",
    "CorrelationCheck",
    "KillSwitch",
    "KillSwitchTrip",
    "Ledger",
    "LimitCheck",
    "RiskDecision",
    "RiskManager",
    "SizingResult",
    "check_concurrent_positions",
    "check_correlation",
    "check_daily_loss",
    "check_daily_trades",
    "check_exposure",
    "check_global_cooldown",
    "check_symbol_cooldown",
    "check_weekly_loss",
    "correlation_matrix",
    "day_start",
    "pairwise_correlation",
    "position_size",
    "returns_from_closes",
    "week_start",
]
