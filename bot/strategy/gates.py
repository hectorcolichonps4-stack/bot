"""Filtros obligatorios.

Un gate responde si o no, nunca "mas o menos": si alguno falla no se llega a
puntuar. Cada uno devuelve el motivo exacto del rechazo para que el registro
sirva despues para diagnosticar por que el bot no esta operando.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

from bot.config.schema import GatesConfig
from bot.execution.base import Side
from bot.strategy.features import MarketSnapshot


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    reason: str = ""
    detail: dict[str, object] | None = None

    @classmethod
    def ok(cls, name: str) -> "GateResult":
        return cls(name=name, passed=True)

    @classmethod
    def fail(cls, name: str, reason: str, **detail: object) -> "GateResult":
        return cls(name=name, passed=False, reason=reason, detail=detail or None)


Gate = Callable[[MarketSnapshot, Side, GatesConfig], GateResult]


def gate_blacklist(snap: MarketSnapshot, side: Side, cfg: GatesConfig) -> GateResult:
    if snap.symbol in {s.upper() for s in cfg.blacklist}:
        return GateResult.fail("blacklist", f"{snap.symbol} esta en la lista negra")
    return GateResult.ok("blacklist")


def gate_history(snap: MarketSnapshot, side: Side, cfg: GatesConfig) -> GateResult:
    if snap.bars_available < cfg.min_bars_available:
        return GateResult.fail(
            "history",
            f"solo {snap.bars_available} velas, se exigen {cfg.min_bars_available}",
            bars=snap.bars_available,
        )
    return GateResult.ok("history")


def gate_side_allowed(snap: MarketSnapshot, side: Side, cfg: GatesConfig) -> GateResult:
    if side is Side.SHORT and not cfg.allow_short:
        return GateResult.fail("side_allowed", "los cortos estan desactivados")
    return GateResult.ok("side_allowed")


def gate_trend(snap: MarketSnapshot, side: Side, cfg: GatesConfig) -> GateResult:
    """Sin alineacion de medias no se opera a favor de tendencia."""
    if not cfg.require_trend_alignment:
        return GateResult.ok("trend")
    aligned = snap.trend_up if side is Side.LONG else snap.trend_down
    if not aligned:
        return GateResult.fail(
            "trend",
            "las EMAs no estan alineadas con el lado de la operacion",
            ema_fast=snap.ema_fast,
            ema_slow=snap.ema_slow,
            ema_trend=snap.ema_trend,
        )
    return GateResult.ok("trend")


def gate_volume(snap: MarketSnapshot, side: Side, cfg: GatesConfig) -> GateResult:
    if snap.relative_volume < cfg.min_relative_volume:
        return GateResult.fail(
            "volume",
            f"volumen relativo {snap.relative_volume:.2f} < {cfg.min_relative_volume}",
            relative_volume=snap.relative_volume,
        )
    if cfg.min_quote_volume and snap.quote_volume < cfg.min_quote_volume:
        return GateResult.fail(
            "liquidity",
            f"volumen en quote {snap.quote_volume:,.0f} < {cfg.min_quote_volume:,.0f}",
            quote_volume=snap.quote_volume,
        )
    return GateResult.ok("volume")


def gate_rsi(snap: MarketSnapshot, side: Side, cfg: GatesConfig) -> GateResult:
    """Evita perseguir un movimiento ya extendido."""
    if side is Side.LONG and snap.rsi > cfg.rsi_long_max:
        return GateResult.fail("rsi", f"RSI {snap.rsi:.1f} sobrecomprado (max {cfg.rsi_long_max})", rsi=snap.rsi)
    if side is Side.SHORT and snap.rsi < cfg.rsi_short_min:
        return GateResult.fail("rsi", f"RSI {snap.rsi:.1f} sobrevendido (min {cfg.rsi_short_min})", rsi=snap.rsi)
    return GateResult.ok("rsi")


def gate_volatility(snap: MarketSnapshot, side: Side, cfg: GatesConfig) -> GateResult:
    """Ni mercado muerto ni mercado ingobernable."""
    if snap.atr_pct < cfg.min_atr_pct:
        return GateResult.fail("volatility", f"ATR {snap.atr_pct:.2f}% por debajo de {cfg.min_atr_pct}%", atr_pct=snap.atr_pct)
    if snap.atr_pct > cfg.max_atr_pct:
        return GateResult.fail("volatility", f"ATR {snap.atr_pct:.2f}% por encima de {cfg.max_atr_pct}%", atr_pct=snap.atr_pct)
    return GateResult.ok("volatility")


def gate_spread(snap: MarketSnapshot, side: Side, cfg: GatesConfig) -> GateResult:
    if snap.spread_pct > cfg.max_spread_pct:
        return GateResult.fail("spread", f"spread {snap.spread_pct:.3f}% > {cfg.max_spread_pct}%", spread_pct=snap.spread_pct)
    return GateResult.ok("spread")


def gate_room_to_target(snap: MarketSnapshot, side: Side, cfg: GatesConfig) -> GateResult:
    """Exige al menos un ATR de recorrido hasta el nivel que estorba."""
    if side is Side.LONG:
        barrier = snap.nearest_resistance
        room = (barrier - snap.close) if barrier else float("inf")
    else:
        barrier = snap.nearest_support
        room = (snap.close - barrier) if barrier else float("inf")
    if snap.atr > 0 and room < snap.atr:
        return GateResult.fail(
            "room_to_target",
            f"solo {room:.4f} de recorrido hasta {barrier:.4f} (ATR {snap.atr:.4f})",
            room=room,
            barrier=barrier,
        )
    return GateResult.ok("room_to_target")


DEFAULT_GATES: tuple[Gate, ...] = (
    gate_blacklist,
    gate_history,
    gate_side_allowed,
    gate_trend,
    gate_volume,
    gate_rsi,
    gate_volatility,
    gate_spread,
    gate_room_to_target,
)


def run_gates(
    snap: MarketSnapshot,
    side: Side,
    cfg: GatesConfig,
    gates: Sequence[Gate] = DEFAULT_GATES,
) -> List[GateResult]:
    """Ejecuta todos los gates y devuelve el resultado de cada uno.

    No corta en el primer fallo a proposito: conocer todos los motivos de
    rechazo a la vez ahorra muchas iteraciones al afinar los parametros.
    """
    if not cfg.enabled:
        return [GateResult.ok("disabled")]
    return [gate(snap, side, cfg) for gate in gates]


def first_failure(results: Sequence[GateResult]) -> GateResult | None:
    for result in results:
        if not result.passed:
            return result
    return None
