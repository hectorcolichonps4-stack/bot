"""Periodos de enfriamiento tras una perdida o tras operar un simbolo.

Sirven para romper el bucle de reentrar inmediatamente en el mismo par en las
mismas condiciones que acaban de fallar.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from bot.config.schema import RiskConfig
from bot.risk.ledger import Ledger


@dataclass(frozen=True)
class CooldownCheck:
    name: str
    passed: bool
    reason: str = ""
    remaining_minutes: float = 0.0


def check_global_cooldown(ledger: Ledger, now: pd.Timestamp, cfg: RiskConfig) -> CooldownCheck:
    """Pausa toda la operativa durante un rato tras cerrar en perdidas."""
    if cfg.cooldown_minutes_after_loss <= 0:
        return CooldownCheck("cooldown_after_loss", True)

    last_loss = ledger.last_loss()
    if last_loss is None:
        return CooldownCheck("cooldown_after_loss", True)

    elapsed = (now - last_loss.closed_at).total_seconds() / 60.0
    remaining = cfg.cooldown_minutes_after_loss - elapsed
    if remaining > 0:
        return CooldownCheck(
            "cooldown_after_loss", False,
            f"enfriamiento tras perdida en {last_loss.symbol}: quedan {remaining:.0f} min",
            remaining,
        )
    return CooldownCheck("cooldown_after_loss", True)


def check_symbol_cooldown(ledger: Ledger, symbol: str, now: pd.Timestamp, cfg: RiskConfig) -> CooldownCheck:
    """Impide reentrar en el mismo simbolo demasiado pronto."""
    if cfg.cooldown_minutes_same_symbol <= 0:
        return CooldownCheck("cooldown_symbol", True)

    last = ledger.last_trade_for(symbol)
    if last is None:
        return CooldownCheck("cooldown_symbol", True)

    elapsed = (now - last.closed_at).total_seconds() / 60.0
    remaining = cfg.cooldown_minutes_same_symbol - elapsed
    if remaining > 0:
        return CooldownCheck(
            "cooldown_symbol", False,
            f"{symbol} en enfriamiento: quedan {remaining:.0f} min",
            remaining,
        )
    return CooldownCheck("cooldown_symbol", True)
