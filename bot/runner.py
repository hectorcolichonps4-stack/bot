"""Bucle de operativa para los modos paper y live.

El ciclo es siempre el mismo y deliberadamente aburrido: sincronizar estado,
gestionar lo que ya esta abierto, buscar oportunidades, pasarlas por riesgo y
solo entonces enviar la orden. Cualquier excepcion cuenta para el kill switch.
"""

from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from bot.config.loader import LoadedConfig
from bot.data.feed import DataFeed
from bot.execution.base import Broker, ExecutionError
from bot.logging.db import Database
from bot.risk.manager import RiskManager
from bot.strategy.engine import StrategyEngine
from bot.strategy.signals import Decision

log = logging.getLogger(__name__)


@dataclass
class SessionStats:
    """Contadores de la sesion, utiles para el resumen final y el dashboard."""

    iterations: int = 0
    signals: int = 0
    rejections: int = 0
    orders: int = 0
    errors: int = 0
    started_at: pd.Timestamp = field(default_factory=lambda: pd.Timestamp.now(tz="UTC"))

    def as_dict(self) -> dict[str, object]:
        return {
            "iteraciones": self.iterations,
            "senales": self.signals,
            "rechazos": self.rejections,
            "ordenes": self.orders,
            "errores": self.errors,
            "inicio": str(self.started_at),
        }


class TradingSession:
    """Orquesta feed, estrategia, riesgo y broker en tiempo real."""

    def __init__(
        self,
        loaded: LoadedConfig,
        *,
        feed: DataFeed,
        broker: Broker,
        db: Database,
        run_id: int,
    ) -> None:
        self.loaded = loaded
        self.config = loaded.config
        self.feed = feed
        self.broker = broker
        self.db = db
        self.run_id = run_id

        self.strategy = StrategyEngine(loaded)
        self.risk = RiskManager(loaded)
        self.stats = SessionStats()
        self._stop = False
        self._closes: Dict[str, pd.Series] = {}

        self.risk.ledger.mark_equity(broker.account().equity)

    # ------------------------------------------------------------------ control

    def request_stop(self, *_: object) -> None:
        """Pide una parada ordenada; el ciclo en curso termina antes de salir."""
        log.info("parada solicitada; se terminara el ciclo en curso")
        self._stop = True

    def install_signal_handlers(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self.request_stop)

    # --------------------------------------------------------------------- ciclo

    def run_forever(self, *, max_iterations: int | None = None) -> SessionStats:
        """Ejecuta ciclos hasta que se pida parar o salte el kill switch."""
        log.info(
            "sesion %s iniciada | simbolos=%s | timeframe=%s | config=%s",
            self.broker.name, ",".join(self.config.data.symbols), self.config.data.timeframe, self.loaded.config_hash,
        )
        while not self._stop:
            if max_iterations is not None and self.stats.iterations >= max_iterations:
                break

            started = time.monotonic()
            try:
                self.run_once()
            except Exception as exc:  # noqa: BLE001 - un ciclo roto no tumba la sesion
                self.stats.errors += 1
                self.risk.on_error()
                log.error("fallo en el ciclo de operativa: %s", exc, exc_info=True)

            if self.risk.kill_switch.active:
                trip = self.risk.kill_switch.trip
                log.error("kill switch activo (%s); se cierran posiciones y se sale", trip.reason if trip else "")
                self._emergency_close()
                break

            elapsed = time.monotonic() - started
            self._sleep(max(0.0, self.config.execution.poll_seconds - elapsed))

        self.db.set_state("last_session", self.stats.as_dict())
        log.info("sesion terminada: %s", self.stats.as_dict())
        return self.stats

    def run_once(self) -> List[Decision]:
        """Un ciclo completo: gestionar abiertas, buscar nuevas, ejecutar."""
        self.stats.iterations += 1
        self.broker.sync()
        self._manage_open_positions()

        account = self.broker.account()
        self.risk.on_equity(account.equity)
        self.db.record_equity(
            equity=account.equity,
            cash=account.cash,
            exposure=account.exposure,
            open_positions=len(account.positions),
            run_id=self.run_id,
        )

        if self.risk.kill_switch.active:
            return []

        decisions = self._scan()
        for decision in decisions:
            if decision.accepted:
                self._handle_signal(decision)
            elif decision.rejection is not None:
                self._log_rejection(decision)
        return decisions

    # ------------------------------------------------------------------ interno

    def _scan(self) -> List[Decision]:
        frames: Dict[str, pd.DataFrame] = {}
        for symbol in self.config.data.symbols:
            try:
                frame = self.feed.latest(symbol)
            except Exception as exc:  # noqa: BLE001 - un simbolo caido no para el resto
                self.stats.errors += 1
                self.risk.on_error()
                log.warning("sin datos para %s: %s", symbol, exc)
                continue
            frames[symbol] = frame
            self._closes[symbol] = frame["close"]
        return self.strategy.scan(frames) if frames else []

    def _handle_signal(self, decision: Decision) -> None:
        signal_obj = decision.signal
        self.stats.signals += 1
        signal_id = self.db.log_signal(signal_obj, run_id=self.run_id)

        spread = self._spread(signal_obj.symbol)
        if spread > self.config.gates.max_spread_pct:
            self._record_rejection(
                signal_obj.symbol, signal_obj.timestamp, "spread",
                f"spread {spread:.3f}% supera {self.config.gates.max_spread_pct}% en el momento de ejecutar",
            )
            return

        verdict = self.risk.evaluate(
            signal_obj,
            self.broker.account(),
            filters=self.broker.filters(signal_obj.symbol),
            closes=self._closes,
            now=pd.Timestamp.now(tz="UTC"),
        )
        if not verdict.approved:
            self._record_rejection(
                signal_obj.symbol, signal_obj.timestamp, "risk", f"[{verdict.stage}] {verdict.reason}"
            )
            return

        try:
            self.broker.open_position(
                signal_obj.symbol,
                signal_obj.side,
                verdict.quantity,
                stop_loss=signal_obj.stop_loss,
                take_profit=signal_obj.take_profit,
                metadata={"signal_id": signal_id, "score": signal_obj.score},
            )
        except ExecutionError as exc:
            self.stats.errors += 1
            self.risk.on_error()
            self._record_rejection(signal_obj.symbol, signal_obj.timestamp, "execution", str(exc))
            return

        self.stats.orders += 1
        self.db.mark_signal_acted(signal_id)
        log.info(
            "orden enviada %s %s qty=%.8f score=%.1f",
            signal_obj.side.value, signal_obj.symbol, verdict.quantity, signal_obj.score,
        )

    def _manage_open_positions(self) -> None:
        """Cierra lo que haya tocado stop u objetivo y actualiza el riesgo."""
        checker = getattr(self.broker, "check_live_exits", None)
        if checker is None:
            return
        for trade in checker():
            self.risk.on_trade_closed(trade, self.broker.account().equity)

    def _emergency_close(self) -> None:
        for position in list(self.broker.positions()):
            try:
                self.broker.close_position(position.symbol, reason="kill_switch")
            except ExecutionError:
                log.error("no se pudo cerrar %s en la parada de emergencia", position.symbol, exc_info=True)

    def _spread(self, symbol: str) -> float:
        try:
            return float(self.feed.ticker(symbol).get("spread_pct", 0.0))
        except Exception:  # noqa: BLE001 - sin libro se asume 0 y deciden los gates
            return 0.0

    def _log_rejection(self, decision: Decision) -> None:
        rejection = decision.rejection
        self._record_rejection(
            rejection.symbol, rejection.timestamp, rejection.stage, rejection.reason, dict(rejection.detail)
        )

    def _record_rejection(
        self,
        symbol: str,
        bar_time: pd.Timestamp,
        stage: str,
        reason: str,
        detail: dict | None = None,
    ) -> None:
        self.stats.rejections += 1
        if not self.config.logging.log_rejections:
            return
        self.db.log_rejection(
            symbol=symbol,
            bar_time=bar_time,
            stage=stage,
            reason=reason,
            detail=detail or {},
            config_hash=self.loaded.config_hash,
            run_id=self.run_id,
        )

    @staticmethod
    def _sleep(seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)
