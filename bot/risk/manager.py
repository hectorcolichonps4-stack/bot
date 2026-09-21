"""Coordinador de riesgo: la ultima palabra antes de enviar una orden.

Ninguna senal llega al broker sin pasar por aqui. El orden de comprobacion va
de lo mas grave a lo mas local: primero lo que detiene todo el bot (kill
switch), luego los limites de cuenta, despues los especificos del simbolo y
por ultimo el dimensionado.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Mapping, Sequence

import pandas as pd

from bot.config.loader import LoadedConfig
from bot.data.filters import SymbolFilters
from bot.execution.base import AccountState, Position
from bot.risk.correlation import CorrelationCheck, check_correlation
from bot.risk.cooldown import CooldownCheck, check_global_cooldown, check_symbol_cooldown
from bot.risk.kill_switch import KillSwitch
from bot.risk.ledger import ClosedTrade, Ledger
from bot.risk.limits import (
    LimitCheck,
    check_concurrent_positions,
    check_daily_loss,
    check_daily_trades,
    check_exposure,
    check_weekly_loss,
)
from bot.risk.sizing import SizingResult, position_size
from bot.strategy.signals import Signal

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RiskDecision:
    """Veredicto de riesgo, con el rastro completo de lo comprobado."""

    approved: bool
    quantity: float = 0.0
    reason: str = ""
    stage: str = ""
    sizing: SizingResult | None = None
    checks: List[str] = field(default_factory=list)

    @property
    def rejected(self) -> bool:
        return not self.approved


class RiskManager:
    """Estado de riesgo vivo de la sesion mas las reglas de admision."""

    def __init__(self, loaded: LoadedConfig, *, ledger: Ledger | None = None) -> None:
        self.loaded = loaded
        self.config = loaded.config.risk
        self.ledger = ledger or Ledger()
        self.kill_switch = KillSwitch(self.config.kill_switch)

    # ----------------------------------------------------------- ciclo de vida

    def on_trade_closed(self, trade: ClosedTrade, equity: float, now: pd.Timestamp | None = None) -> None:
        """Actualiza el historial y reevalua el kill switch tras cada cierre."""
        self.ledger.record(trade)
        self.ledger.mark_equity(equity)
        trip = self.kill_switch.evaluate(self.ledger, equity, now or trade.closed_at)
        if trip is not None:
            log.error("KILL SWITCH activado: %s", trip.reason)

    def on_equity(self, equity: float, now: pd.Timestamp | None = None) -> None:
        """Marca el nuevo maximo de equity y vigila el drawdown en curso."""
        self.ledger.mark_equity(equity)
        self.kill_switch.evaluate(self.ledger, equity, now or pd.Timestamp.now(tz="UTC"))

    def on_error(self, now: pd.Timestamp | None = None) -> None:
        self.kill_switch.record_error(now)

    def rearm_kill_switch(self, equity: float) -> None:
        """Rearma el interruptor, como haria un operador tras revisar la sesion.

        Se limpia la racha y se rebaja la marca de maximo al capital actual:
        de lo contrario las mismas condiciones que dispararon el freno lo
        volverian a disparar en el ciclo siguiente.
        """
        self.kill_switch.reset()
        self.ledger.reset_streak()
        self.ledger.equity_high_water = equity
        log.warning("kill switch rearmado con equity %.2f", equity)

    # ------------------------------------------------------------- evaluacion

    def evaluate(
        self,
        signal: Signal,
        account: AccountState,
        *,
        filters: SymbolFilters,
        closes: Mapping[str, pd.Series] | None = None,
        now: pd.Timestamp | None = None,
    ) -> RiskDecision:
        """Aprueba o rechaza la senal y, si la aprueba, dice con que cantidad."""
        now = now or signal.timestamp
        passed: List[str] = []

        trip = self.kill_switch.evaluate(self.ledger, account.equity, now)
        if trip is not None:
            return RiskDecision(False, reason=f"kill switch activo: {trip.reason}", stage="kill_switch", checks=passed)
        passed.append("kill_switch")

        if self._already_open(signal.symbol, account.positions):
            return RiskDecision(False, reason=f"ya hay una posicion abierta en {signal.symbol}", stage="duplicate", checks=passed)
        passed.append("duplicate")

        account_checks: Sequence[LimitCheck] = (
            check_daily_loss(self.ledger, account.equity, now, self.config),
            check_weekly_loss(self.ledger, account.equity, now, self.config),
            check_daily_trades(self.ledger, now, self.config),
            check_concurrent_positions(len(account.positions), self.config),
        )
        for check in account_checks:
            if not check.passed:
                return RiskDecision(False, reason=check.reason, stage=check.name, checks=passed)
            passed.append(check.name)

        cooldowns: Sequence[CooldownCheck] = (
            check_global_cooldown(self.ledger, now, self.config),
            check_symbol_cooldown(self.ledger, signal.symbol, now, self.config),
        )
        for cooldown in cooldowns:
            if not cooldown.passed:
                return RiskDecision(False, reason=cooldown.reason, stage=cooldown.name, checks=passed)
            passed.append(cooldown.name)

        correlation: CorrelationCheck = check_correlation(
            signal.symbol,
            [p.symbol for p in account.positions],
            closes or {},
            lookback=self.config.correlation_lookback,
            max_correlation=self.config.max_correlation,
        )
        if not correlation.passed:
            return RiskDecision(False, reason=correlation.reason, stage="correlation", checks=passed)
        passed.append("correlation")

        sizing = position_size(
            equity=account.equity,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            risk_per_trade_pct=self.config.risk_per_trade_pct,
            max_position_pct=self.config.max_position_pct,
            available_cash=account.cash,
            filters=filters,
        )
        if not sizing.viable:
            return RiskDecision(False, reason=sizing.reason, stage="sizing", sizing=sizing, checks=passed)
        passed.append("sizing")

        new_exposure_pct = sizing.notional / account.equity * 100.0 if account.equity > 0 else 0.0
        exposure = check_exposure(account.exposure_pct, new_exposure_pct, self.config)
        if not exposure.passed:
            return RiskDecision(False, reason=exposure.reason, stage="exposure", sizing=sizing, checks=passed)
        passed.append("exposure")

        return RiskDecision(
            approved=True,
            quantity=sizing.quantity,
            reason=sizing.reason or "aprobado",
            stage="approved",
            sizing=sizing,
            checks=passed,
        )

    # ---------------------------------------------------------------- interno

    @staticmethod
    def _already_open(symbol: str, positions: Sequence[Position]) -> bool:
        return any(p.symbol == symbol.upper() for p in positions)
