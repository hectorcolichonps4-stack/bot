"""Limites diarios y semanales de perdida y de numero de operaciones."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from bot.config.schema import RiskConfig
from bot.risk.ledger import Ledger, day_start, week_start


@dataclass(frozen=True)
class LimitCheck:
    name: str
    passed: bool
    reason: str = ""
    value: float = 0.0
    limit: float = 0.0


def check_daily_loss(ledger: Ledger, equity: float, now: pd.Timestamp, cfg: RiskConfig) -> LimitCheck:
    """Perdida acumulada hoy frente al maximo permitido.

    El limite se mide sobre el equity actual, no sobre el del inicio del dia:
    asi una racha mala reduce progresivamente el tamano de lo que se arriesga.
    """
    pnl = ledger.pnl_since(day_start(now))
    limit = equity * cfg.max_daily_loss_pct / 100.0
    if pnl < 0 and abs(pnl) >= limit:
        return LimitCheck(
            "daily_loss", False,
            f"perdida diaria {abs(pnl):.2f} alcanza el limite {limit:.2f} ({cfg.max_daily_loss_pct}%)",
            abs(pnl), limit,
        )
    return LimitCheck("daily_loss", True, value=abs(min(pnl, 0.0)), limit=limit)


def check_weekly_loss(ledger: Ledger, equity: float, now: pd.Timestamp, cfg: RiskConfig) -> LimitCheck:
    pnl = ledger.pnl_since(week_start(now))
    limit = equity * cfg.max_weekly_loss_pct / 100.0
    if pnl < 0 and abs(pnl) >= limit:
        return LimitCheck(
            "weekly_loss", False,
            f"perdida semanal {abs(pnl):.2f} alcanza el limite {limit:.2f} ({cfg.max_weekly_loss_pct}%)",
            abs(pnl), limit,
        )
    return LimitCheck("weekly_loss", True, value=abs(min(pnl, 0.0)), limit=limit)


def check_daily_trades(ledger: Ledger, now: pd.Timestamp, cfg: RiskConfig) -> LimitCheck:
    count = ledger.count_since(day_start(now))
    if count >= cfg.max_daily_trades:
        return LimitCheck(
            "daily_trades", False,
            f"{count} operaciones hoy, el maximo es {cfg.max_daily_trades}",
            count, cfg.max_daily_trades,
        )
    return LimitCheck("daily_trades", True, value=count, limit=cfg.max_daily_trades)


def check_concurrent_positions(open_positions: int, cfg: RiskConfig) -> LimitCheck:
    if open_positions >= cfg.max_concurrent_positions:
        return LimitCheck(
            "concurrent_positions", False,
            f"{open_positions} posiciones abiertas, el maximo es {cfg.max_concurrent_positions}",
            open_positions, cfg.max_concurrent_positions,
        )
    return LimitCheck("concurrent_positions", True, value=open_positions, limit=cfg.max_concurrent_positions)


def check_exposure(current_exposure_pct: float, new_notional_pct: float, cfg: RiskConfig) -> LimitCheck:
    total = current_exposure_pct + new_notional_pct
    if total > cfg.max_exposure_pct:
        return LimitCheck(
            "exposure", False,
            f"exposicion resultante {total:.1f}% supera el maximo {cfg.max_exposure_pct}%",
            total, cfg.max_exposure_pct,
        )
    return LimitCheck("exposure", True, value=total, limit=cfg.max_exposure_pct)
