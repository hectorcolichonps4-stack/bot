"""Motor de backtest por eventos.

Recorre la linea temporal barra a barra, en el mismo orden en que ocurren las
cosas en vivo:

1. se ejecutan las entradas decididas en la barra anterior, a la apertura;
2. se comprueban stops y objetivos contra el rango de la barra;
3. se cierra la barra y se anota el capital;
4. se evaluan senales nuevas con la barra ya cerrada.

Separar (4) de (1) es lo que impide el sesgo de anticipacion: ninguna decision
se ejecuta al precio que la origino.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence

import pandas as pd

from bot.config.loader import LoadedConfig
from bot.execution.backtest import BacktestBroker
from bot.execution.base import ExecutionError
from bot.logging.db import Database
from bot.backtesting.metrics import Metrics, compute_metrics, trades_to_frame
from bot.risk.ledger import ClosedTrade, Ledger
from bot.risk.manager import RiskManager
from bot.strategy.engine import StrategyEngine
from bot.strategy.signals import Rejection, Signal

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Todo lo que produce una corrida: curva, operaciones, metricas y rechazos."""

    equity_curve: pd.Series
    trades: List[ClosedTrade]
    metrics: Metrics
    rejections: List[Rejection] = field(default_factory=list)
    signals: List[Signal] = field(default_factory=list)
    config_hash: str = ""
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    halted_reason: str = ""
    """Motivo por el que la corrida se detuvo antes de tiempo (kill switch)."""
    kill_switch_events: List[tuple] = field(default_factory=list)
    """Pares (instante, motivo) de cada activacion del freno durante la corrida."""

    @property
    def trades_frame(self) -> pd.DataFrame:
        return trades_to_frame(self.trades)

    def rejection_counts(self) -> pd.Series:
        """Cuantas veces rechazo cada motivo: el primer sitio donde mirar."""
        if not self.rejections:
            return pd.Series(dtype="int64")
        reasons = [f"{r.stage}: {r.reason.split(']')[0].strip('[')}" for r in self.rejections]
        return pd.Series(reasons).value_counts()


@dataclass(frozen=True)
class _PendingEntry:
    """Senal aprobada esperando a ejecutarse en la apertura siguiente."""

    signal: Signal
    quantity: float


class BacktestEngine:
    """Ata estrategia, riesgo y broker simulado sobre datos historicos."""

    def __init__(
        self,
        loaded: LoadedConfig,
        *,
        db: Database | None = None,
        run_id: int | None = None,
        record_rejections: bool = True,
        on_kill_switch: str = "resume_next_day",
    ) -> None:
        """``on_kill_switch`` decide que pasa cuando salta el freno.

        ``resume_next_day`` (por defecto) reanuda al dia siguiente, que es lo
        que haria un operador al revisar la sesion, y permite evaluar todo el
        periodo. ``halt`` corta la corrida ahi: sirve para ver en que momento
        la estrategia habria obligado a parar.
        """
        if on_kill_switch not in ("resume_next_day", "halt"):
            raise ValueError(f"on_kill_switch invalido: {on_kill_switch}")
        self.loaded = loaded
        self.config = loaded.config
        self.db = db
        self.run_id = run_id
        self.record_rejections = record_rejections
        self.on_kill_switch = on_kill_switch

    def run(
        self,
        frames: Mapping[str, pd.DataFrame],
        *,
        initial_balance: float | None = None,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        prepared: Mapping[str, pd.DataFrame] | None = None,
    ) -> BacktestResult:
        """Ejecuta el backtest sobre las velas dadas, indexadas por simbolo.

        ``prepared`` permite reutilizar cuadros de indicadores ya calculados
        (lo aprovecha el walk-forward, que corre muchas veces sobre el mismo
        historico) en vez de recalcularlos en cada corrida.
        """
        balance = initial_balance if initial_balance is not None else self.config.backtesting.initial_balance
        strategy = StrategyEngine(self.loaded)
        risk = RiskManager(self.loaded, ledger=Ledger())
        broker = BacktestBroker(
            initial_balance=balance,
            fee_bps=self.config.execution.fee_bps,
            slippage_bps=self.config.execution.slippage_bps,
            config_hash=self.loaded.config_hash,
        )
        risk.ledger.mark_equity(balance)

        features = {symbol: strategy.prepare(frame) for symbol, frame in frames.items() if not frame.empty}
        features = {symbol: _clip(frame, start, end) for symbol, frame in features.items()}
        features = {symbol: frame for symbol, frame in features.items() if not frame.empty}
        if not features:
            raise ValueError("no hay velas en el rango solicitado")

        # Mapa timestamp -> posicion por simbolo, mas las columnas OHLC como
        # arrays: acceder fila a fila con pandas costaba mas que evaluar la
        # estrategia entera.
        offsets = {
            symbol: {timestamp: i for i, timestamp in enumerate(frame["open_time"])}
            for symbol, frame in features.items()
        }
        ohlc = {
            symbol: {name: frame[name].to_numpy(dtype="float64") for name in ("open", "high", "low", "close")}
            for symbol, frame in features.items()
        }
        builders = {symbol: strategy.builder(symbol, frame) for symbol, frame in features.items()}
        timeline = sorted(set().union(*[set(offset) for offset in offsets.values()]))

        equity_points: Dict[pd.Timestamp, float] = {}
        rejections: List[Rejection] = []
        signals: List[Signal] = []
        pending: List[_PendingEntry] = []
        halted_reason = ""
        kill_events: List[tuple] = []
        last_timestamp = timeline[0] if timeline else None

        for timestamp in timeline:
            last_timestamp = timestamp
            here = {symbol: offset[timestamp] for symbol, offset in offsets.items() if timestamp in offset}
            bars = {
                symbol: {name: column[index] for name, column in ohlc[symbol].items()}
                for symbol, index in here.items()
            }
            broker.now = timestamp

            pending = self._execute_pending(broker, risk, pending, bars)
            self._apply_market(broker, bars, use_close=False)

            for trade in broker.check_exits(_ranges(bars)):
                risk.on_trade_closed(trade, broker.equity(), timestamp)

            self._apply_market(broker, bars, use_close=True)
            equity = broker.equity()
            equity_points[timestamp] = equity
            risk.on_equity(equity, timestamp)

            if risk.kill_switch.active:
                trip = risk.kill_switch.trip
                reason = trip.reason if trip else "kill switch"
                if trip is not None and (not kill_events or kill_events[-1][0] != trip.triggered_at):
                    kill_events.append((trip.triggered_at, reason))

                if self.on_kill_switch == "halt":
                    halted_reason = reason
                    log.warning("backtest detenido en %s: %s", timestamp, reason)
                    break

                # Se reanuda al dia siguiente: el freno cuesta las operaciones
                # del resto de la jornada, que es su coste real, pero no
                # invalida el resto del periodo estudiado.
                if trip is not None and timestamp.normalize() > trip.triggered_at.normalize():
                    risk.rearm_kill_switch(equity)
                else:
                    continue

            pending.extend(
                self._collect_signals(strategy, risk, broker, builders, features, here, rejections, signals)
            )

        broker.close_all(reason="kill_switch" if halted_reason else "backtest_end")
        if last_timestamp is not None:
            equity_points[last_timestamp] = broker.equity()

        equity_curve = pd.Series(equity_points).sort_index()
        equity_curve.index = pd.to_datetime(equity_curve.index)

        result = BacktestResult(
            equity_curve=equity_curve,
            trades=list(broker.closed_trades),
            metrics=compute_metrics(equity_curve, broker.closed_trades),
            rejections=rejections,
            signals=signals,
            config_hash=self.loaded.config_hash,
            start=timeline[0] if timeline else None,
            end=last_timestamp,
            halted_reason=halted_reason,
            kill_switch_events=kill_events,
        )
        self._persist(result)
        return result

    # ---------------------------------------------------------------- interno

    def _execute_pending(
        self,
        broker: BacktestBroker,
        risk: RiskManager,
        pending: Sequence[_PendingEntry],
        bars: Mapping[str, Mapping[str, float]],
    ) -> List[_PendingEntry]:
        """Abre en la apertura de esta barra lo aprobado en la anterior."""
        carried: List[_PendingEntry] = []
        for entry in pending:
            bar = bars.get(entry.signal.symbol)
            if bar is None:
                carried.append(entry)  # el simbolo no cotiza aqui; se intenta en la siguiente
                continue
            broker.set_price(entry.signal.symbol, float(bar["open"]))
            try:
                broker.open_position(
                    entry.signal.symbol,
                    entry.signal.side,
                    entry.quantity,
                    stop_loss=entry.signal.stop_loss,
                    take_profit=entry.signal.take_profit,
                    metadata={"score": entry.signal.score},
                )
            except ExecutionError as exc:
                log.debug("entrada descartada en %s: %s", entry.signal.symbol, exc)
                risk.on_error(broker.now)
        return carried

    @staticmethod
    def _apply_market(broker: BacktestBroker, bars: Mapping[str, Mapping[str, float]], *, use_close: bool) -> None:
        column = "close" if use_close else "open"
        for symbol, bar in bars.items():
            broker.set_price(symbol, float(bar[column]))

    def _collect_signals(
        self,
        strategy: StrategyEngine,
        risk: RiskManager,
        broker: BacktestBroker,
        builders: Mapping[str, object],
        features: Mapping[str, pd.DataFrame],
        here: Mapping[str, int],
        rejections: List[Rejection],
        signals: List[Signal],
    ) -> List[_PendingEntry]:
        """Evalua cada simbolo sobre la barra recien cerrada."""
        approved: List[_PendingEntry] = []
        account = broker.account()
        open_symbols = {p.symbol for p in account.positions}

        for symbol, index in here.items():
            if symbol in open_symbols:
                continue

            decision = strategy.evaluate_at(builders[symbol], index)

            if not decision.accepted:
                if decision.rejection is not None:
                    rejections.append(decision.rejection)
                continue

            signal = decision.signal
            signals.append(signal)

            # Solo las velas ya vistas entran en la correlacion.
            closes = {
                other: features[other]["close"].iloc[: position + 1]
                for other, position in here.items()
            }
            verdict = risk.evaluate(signal, account, filters=broker.filters(symbol), closes=closes)
            if not verdict.approved:
                rejections.append(
                    Rejection(
                        symbol=symbol,
                        timestamp=signal.timestamp,
                        stage="risk",
                        reason=f"[{verdict.stage}] {verdict.reason}",
                        detail={"score": signal.score},
                    )
                )
                continue

            approved.append(_PendingEntry(signal=signal, quantity=verdict.quantity))
            # Se refresca la cuenta para que la senal siguiente de esta misma
            # barra vea el hueco ya reservado y no duplique exposicion.
            open_symbols.add(symbol)
        return approved

    def _persist(self, result: BacktestResult) -> None:
        if self.db is None:
            return
        for signal in result.signals:
            self.db.log_signal(signal, run_id=self.run_id)
        if self.record_rejections and self.config.logging.log_rejections:
            for rejection in result.rejections:
                self.db.log_rejection(
                    symbol=rejection.symbol,
                    bar_time=rejection.timestamp,
                    stage=rejection.stage,
                    reason=rejection.reason,
                    detail=dict(rejection.detail),
                    config_hash=result.config_hash,
                    run_id=self.run_id,
                )
        for trade in result.trades:
            trade_id = self.db.open_trade(
                symbol=trade.symbol,
                side=trade.side.value,
                quantity=trade.quantity,
                entry_price=trade.entry_price,
                stop_loss=trade.stop_loss,
                take_profit=trade.take_profit,
                opened_at=trade.opened_at,
                mode="backtest",
                config_hash=result.config_hash,
                run_id=self.run_id,
            )
            self.db.close_trade(
                trade_id,
                exit_price=trade.exit_price,
                closed_at=trade.closed_at,
                pnl=trade.pnl,
                fees=trade.fees,
                r_multiple=trade.r_multiple,
                exit_reason=trade.exit_reason,
            )
        for timestamp, equity in result.equity_curve.items():
            self.db.record_equity(equity=float(equity), cash=float(equity), recorded_at=timestamp, run_id=self.run_id)


def _ranges(bars: Mapping[str, Mapping[str, float]]) -> Dict[str, Dict[str, float]]:
    return {symbol: {"high": bar["high"], "low": bar["low"]} for symbol, bar in bars.items()}


def _clip(frame: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.DataFrame:
    out = frame
    if start is not None:
        out = out[out["open_time"] >= _utc(start)]
    if end is not None:
        out = out[out["open_time"] <= _utc(end)]
    return out.reset_index(drop=True)


def _utc(value: pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
