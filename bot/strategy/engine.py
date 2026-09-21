"""Motor de estrategia: velas -> gates -> puntuacion -> senal o rechazo."""

from __future__ import annotations

import logging
import pandas as pd

from bot.config.loader import LoadedConfig
from bot.execution.base import Side
from bot.strategy.features import (
    IncompleteSnapshot,
    MarketSnapshot,
    SnapshotBuilder,
    build_features,
)
from bot.strategy.gates import first_failure, run_gates
from bot.strategy.scoring import Scorer
from bot.strategy.signals import Decision, Rejection, Signal

log = logging.getLogger(__name__)


class StrategyEngine:
    """Decide, para un simbolo y una vela, si hay oportunidad.

    El motor no conoce el broker ni el saldo: solo dice "esto parece una
    oportunidad, con este stop y este objetivo". Cuanto arriesgar, y si se
    puede arriesgar, lo decide ``/risk``.
    """

    def __init__(self, loaded: LoadedConfig) -> None:
        self.loaded = loaded
        self.config = loaded.config
        self.scorer = Scorer(self.config.scoring)

    # ------------------------------------------------------------------ API

    def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Calcula los indicadores una sola vez por simbolo."""
        return build_features(frame, self.config.indicators)

    def sides(self) -> tuple[Side, ...]:
        return (Side.LONG, Side.SHORT) if self.config.gates.allow_short else (Side.LONG,)

    def builder(self, symbol: str, features: pd.DataFrame) -> SnapshotBuilder:
        """Prepara el extractor de estados de un simbolo (una vez por corrida)."""
        return SnapshotBuilder(symbol, features, self.config.indicators)

    def evaluate(
        self,
        symbol: str,
        features: pd.DataFrame,
        *,
        index: int = -1,
        spread_pct: float = 0.0,
    ) -> Decision:
        """Evalua la barra ``index`` del cuadro de indicadores ya calculado."""
        try:
            builder = self.builder(symbol, features)
        except IncompleteSnapshot as exc:
            return Decision.reject(
                Rejection(symbol=symbol.upper(), timestamp=pd.Timestamp.now(tz="UTC"), stage="data", reason=str(exc))
            )
        return self.evaluate_at(builder, index, spread_pct=spread_pct)

    def evaluate_at(self, builder: SnapshotBuilder, index: int = -1, *, spread_pct: float = 0.0) -> Decision:
        """Igual que ``evaluate`` pero reutilizando un builder ya construido."""
        try:
            snap = builder.at(index, spread_pct=spread_pct)
        except IncompleteSnapshot as exc:
            position = index if index >= 0 else builder.length + index
            position = min(max(position, 0), builder.length - 1)
            return Decision.reject(
                Rejection(
                    symbol=builder.symbol,
                    timestamp=pd.Timestamp(builder.open_time[position]),
                    stage="data",
                    reason=str(exc),
                )
            )

        best: Decision | None = None
        rejections: list[Rejection] = []

        for side in self.sides():
            decision = self._evaluate_side(snap, side)
            if decision.accepted:
                if best is None or decision.signal.score > best.signal.score:
                    best = decision
            elif decision.rejection is not None:
                rejections.append(decision.rejection)

        if best is not None:
            return best
        # Con ambos lados evaluados se reporta el rechazo del lado principal.
        return Decision.reject(rejections[0])

    def evaluate_frame(self, symbol: str, frame: pd.DataFrame, **kwargs) -> Decision:
        """Atajo: calcula indicadores y evalua la ultima vela."""
        return self.evaluate(symbol, self.prepare(frame), **kwargs)

    def scan(self, frames: dict[str, pd.DataFrame]) -> list[Decision]:
        """Evalua varios simbolos y devuelve las decisiones ordenadas por score."""
        decisions = [self.evaluate_frame(symbol, frame) for symbol, frame in frames.items()]
        decisions.sort(key=lambda d: d.signal.score if d.signal else -1.0, reverse=True)
        return decisions

    # -------------------------------------------------------------- interno

    def _evaluate_side(self, snap: MarketSnapshot, side: Side) -> Decision:
        results = run_gates(snap, side, self.config.gates)
        failure = first_failure(results)
        if failure is not None:
            failed = [r.name for r in results if not r.passed]
            return Decision.reject(
                Rejection(
                    symbol=snap.symbol,
                    timestamp=snap.timestamp,
                    stage="gate",
                    reason=f"[{failure.name}] {failure.reason}",
                    detail={"side": side.value, "failed_gates": failed, **(failure.detail or {})},
                )
            )

        breakdown = self.scorer.score(snap, side)
        if breakdown.total < self.config.scoring.min_score:
            return Decision.reject(
                Rejection(
                    symbol=snap.symbol,
                    timestamp=snap.timestamp,
                    stage="score",
                    reason=f"score {breakdown.total:.1f} < minimo {self.config.scoring.min_score}",
                    detail={"side": side.value, "components": dict(breakdown.components)},
                )
            )

        stop_loss, take_profit = self._exit_levels(snap, side)
        try:
            signal = Signal(
                symbol=snap.symbol,
                side=side,
                timestamp=snap.timestamp,
                entry_price=snap.close,
                stop_loss=stop_loss,
                take_profit=take_profit,
                score=breakdown.total,
                components=dict(breakdown.components),
                snapshot=snap.as_dict(),
                config_hash=self.loaded.config_hash,
            )
        except ValueError as exc:
            return Decision.reject(
                Rejection(
                    symbol=snap.symbol,
                    timestamp=snap.timestamp,
                    stage="levels",
                    reason=f"niveles de salida incoherentes: {exc}",
                    detail={"side": side.value, "stop": stop_loss, "target": take_profit},
                )
            )
        return Decision.accept(signal)

    def _exit_levels(self, snap: MarketSnapshot, side: Side) -> tuple[float, float]:
        """Stop a N ATR y objetivo a R multiplos de esa distancia.

        Anclar el stop al ATR y el objetivo a un multiplo del riesgo mantiene
        constante la relacion beneficio/riesgo sea cual sea la volatilidad del
        par, que es lo que hace comparables los resultados entre simbolos.
        """
        risk = snap.atr * self.config.risk.stop_atr_multiple
        reward = risk * self.config.risk.take_profit_r_multiple
        if side is Side.LONG:
            return snap.close - risk, snap.close + reward
        return snap.close + risk, snap.close - reward
