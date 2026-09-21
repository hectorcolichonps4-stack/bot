"""Sistema de puntuacion.

Cada componente devuelve un valor en [0, 1] que mide "como de buena" es esa
faceta de la oportunidad. El total se pondera con los pesos de la
configuracion y se normaliza a 100 puntos, de forma que cambiar los pesos
nunca cambia la escala del umbral ``min_score``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping

from bot.config.schema import ScoringConfig
from bot.execution.base import Side
from bot.strategy.features import MarketSnapshot

Component = Callable[[MarketSnapshot, Side], float]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def score_trend(snap: MarketSnapshot, side: Side) -> float:
    """Separacion de las medias, medida en ATR: mas separacion, mas tendencia."""
    if snap.atr <= 0:
        return 0.0
    gap = (snap.ema_fast - snap.ema_slow) * side.sign / snap.atr
    return _clamp01(gap / 2.0)


def score_momentum(snap: MarketSnapshot, side: Side) -> float:
    """Histograma MACD a favor y creciendo."""
    if snap.atr <= 0:
        return 0.0
    strength = _clamp01((snap.macd_hist * side.sign) / (snap.atr * 0.5))
    expanding = (snap.macd_hist - snap.macd_hist_prev) * side.sign > 0
    return strength * (1.0 if expanding else 0.6)


def score_rsi(snap: MarketSnapshot, side: Side) -> float:
    """Premia la zona media-alta (45-65 en largo): con fuerza pero sin agotarse."""
    value = snap.rsi if side is Side.LONG else 100.0 - snap.rsi
    if value < 40.0:
        return _clamp01((value - 25.0) / 15.0) * 0.5
    if value <= 65.0:
        return 1.0
    return _clamp01((80.0 - value) / 15.0)


def score_volume(snap: MarketSnapshot, side: Side) -> float:
    """Saturado en 3x: mas volumen que eso ya no aporta informacion util."""
    return _clamp01((snap.relative_volume - 1.0) / 2.0)


def score_structure(snap: MarketSnapshot, side: Side) -> float:
    """Recorrido libre hasta el siguiente nivel, en multiplos de ATR."""
    if snap.atr <= 0:
        return 0.0
    if side is Side.LONG:
        barrier = snap.nearest_resistance
        room = (barrier - snap.close) if barrier else snap.atr * 4.0
    else:
        barrier = snap.nearest_support
        room = (snap.close - barrier) if barrier else snap.atr * 4.0
    return _clamp01(room / (snap.atr * 3.0))


def score_volatility(snap: MarketSnapshot, side: Side) -> float:
    """Optimo alrededor del 1.5% de ATR: operable sin ser erratico."""
    optimal = 1.5
    distance = abs(snap.atr_pct - optimal) / optimal
    return _clamp01(1.0 - distance / 2.0)


DEFAULT_COMPONENTS: Dict[str, Component] = {
    "trend": score_trend,
    "momentum": score_momentum,
    "rsi": score_rsi,
    "volume": score_volume,
    "structure": score_structure,
    "volatility": score_volatility,
}


@dataclass(frozen=True)
class ScoreBreakdown:
    """Puntuacion total junto al detalle que la explica."""

    total: float
    components: Mapping[str, float]
    weighted: Mapping[str, float]

    def top_contributors(self, count: int = 3) -> list[tuple[str, float]]:
        return sorted(self.weighted.items(), key=lambda item: item[1], reverse=True)[:count]


class Scorer:
    """Combina los componentes segun los pesos configurados."""

    def __init__(self, config: ScoringConfig, components: Mapping[str, Component] | None = None) -> None:
        self.config = config
        self.components = dict(components or DEFAULT_COMPONENTS)

        unknown = set(config.weights) - set(self.components)
        if unknown:
            raise ValueError(f"pesos sin componente asociado: {sorted(unknown)}")

    def score(self, snap: MarketSnapshot, side: Side) -> ScoreBreakdown:
        weights = self.config.weights
        total_weight = sum(weights.values())

        raw: Dict[str, float] = {}
        weighted: Dict[str, float] = {}
        for name, weight in weights.items():
            value = _clamp01(float(self.components[name](snap, side)))
            raw[name] = value
            weighted[name] = value * weight / total_weight * 100.0

        return ScoreBreakdown(total=sum(weighted.values()), components=raw, weighted=weighted)
