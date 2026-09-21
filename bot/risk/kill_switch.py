"""Interruptor de emergencia.

Se activa solo y no se desactiva solo: hace falta una intervencion explicita
(``reset``). Esa asimetria es intencionada, porque las condiciones que lo
disparan son justo aquellas en las que un bot no deberia decidir por si mismo
que ya puede seguir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd

from bot.config.schema import KillSwitchConfig
from bot.risk.ledger import Ledger


@dataclass(frozen=True)
class KillSwitchTrip:
    reason: str
    triggered_at: pd.Timestamp
    detail: dict[str, object] = field(default_factory=dict)


class KillSwitch:
    """Vigila racha de perdidas, drawdown y tasa de errores."""

    def __init__(self, config: KillSwitchConfig) -> None:
        self.config = config
        self.trip: KillSwitchTrip | None = None
        self._error_times: List[pd.Timestamp] = []

    @property
    def active(self) -> bool:
        return self.trip is not None

    def record_error(self, moment: pd.Timestamp | None = None) -> None:
        """Registra un error operativo (fallo de red, orden rechazada, ...)."""
        self._error_times.append(moment or pd.Timestamp.now(tz="UTC"))

    def errors_last_hour(self, now: pd.Timestamp) -> int:
        if not self._error_times:
            return 0
        cutoff = now - pd.Timedelta(hours=1)
        self._error_times = [t for t in self._error_times if t >= cutoff]
        return len(self._error_times)

    def evaluate(self, ledger: Ledger, equity: float, now: pd.Timestamp) -> KillSwitchTrip | None:
        """Comprueba las condiciones de parada. Devuelve el disparo si lo hay."""
        if not self.config.enabled or self.active:
            return self.trip

        streak = ledger.consecutive_losses()
        if streak >= self.config.consecutive_losses:
            self._fire(f"{streak} perdidas consecutivas (limite {self.config.consecutive_losses})", now, streak=streak)
            return self.trip

        drawdown = ledger.drawdown_pct(equity)
        if drawdown >= self.config.max_drawdown_pct:
            self._fire(
                f"drawdown {drawdown:.2f}% alcanza el maximo {self.config.max_drawdown_pct}%",
                now,
                drawdown_pct=drawdown,
                high_water=ledger.equity_high_water,
            )
            return self.trip

        errors = self.errors_last_hour(now)
        if errors >= self.config.max_errors_per_hour:
            self._fire(f"{errors} errores en la ultima hora (limite {self.config.max_errors_per_hour})", now, errors=errors)
            return self.trip

        return None

    def force(self, reason: str, now: pd.Timestamp | None = None) -> KillSwitchTrip:
        """Parada manual, por ejemplo desde el orquestador ante una excepcion."""
        self._fire(reason, now or pd.Timestamp.now(tz="UTC"), manual=True)
        return self.trip  # type: ignore[return-value]

    def reset(self) -> None:
        """Rearme explicito: solo deberia invocarlo un operador humano."""
        self.trip = None
        self._error_times.clear()

    def _fire(self, reason: str, now: pd.Timestamp, **detail: object) -> None:
        self.trip = KillSwitchTrip(reason=reason, triggered_at=now, detail=dict(detail))
