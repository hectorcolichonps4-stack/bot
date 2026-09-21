"""Deteccion de soportes y resistencias por pivotes fractales."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import pandas as pd

from bot.indicators._validate import require_ohlc


@dataclass(frozen=True)
class Levels:
    """Niveles vigentes alrededor del precio de referencia."""

    supports: tuple[float, ...]
    resistances: tuple[float, ...]

    @property
    def nearest_support(self) -> float | None:
        return self.supports[-1] if self.supports else None

    @property
    def nearest_resistance(self) -> float | None:
        return self.resistances[0] if self.resistances else None


def swing_highs(frame: pd.DataFrame, window: int = 3) -> pd.Series:
    """Marca los maximos que superan a sus ``window`` vecinos por cada lado."""
    require_ohlc(frame, ("high",))
    high = frame["high"]
    rolling_max = high.rolling(window=2 * window + 1, center=True, min_periods=2 * window + 1).max()
    return high.where(high == rolling_max).rename("swing_high")


def swing_lows(frame: pd.DataFrame, window: int = 3) -> pd.Series:
    """Marca los minimos que quedan por debajo de sus vecinos."""
    require_ohlc(frame, ("low",))
    low = frame["low"]
    rolling_min = low.rolling(window=2 * window + 1, center=True, min_periods=2 * window + 1).min()
    return low.where(low == rolling_min).rename("swing_low")


def _cluster(values: List[float], tolerance_pct: float) -> List[float]:
    """Agrupa niveles a menos de ``tolerance_pct`` y los resume por su media.

    Dos pivotes casi al mismo precio son el mismo nivel; tratarlos por separado
    inflaria artificialmente la cuenta de toques.
    """
    if not values:
        return []
    ordered = sorted(values)
    clusters: List[List[float]] = [[ordered[0]]]
    for value in ordered[1:]:
        anchor = clusters[-1][0]
        if anchor > 0 and abs(value - anchor) / anchor * 100.0 <= tolerance_pct:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    # Media en Python puro: son grupos de dos o tres valores y el backtest
    # llama aqui cientos de miles de veces; numpy solo anade sobrecarga.
    return [sum(group) / len(group) for group in clusters]


def levels_from_pivots(
    pivots: Iterable[float],
    *,
    reference_price: float,
    tolerance_pct: float = 0.4,
) -> Levels:
    """Agrupa pivotes ya detectados y los reparte a cada lado del precio.

    Se separa de ``support_resistance`` para poder calcular los pivotes una
    sola vez por simbolo y reutilizarlos en cada barra del backtest.
    """
    levels = _cluster([float(v) for v in pivots], tolerance_pct)
    return Levels(
        supports=tuple(level for level in levels if level < reference_price),
        resistances=tuple(level for level in levels if level > reference_price),
    )


def support_resistance(
    frame: pd.DataFrame,
    *,
    lookback: int = 120,
    pivot_window: int = 3,
    tolerance_pct: float = 0.4,
    reference_price: float | None = None,
) -> Levels:
    """Devuelve soportes (por debajo) y resistencias (por encima) del precio.

    Los soportes salen ordenados de lejos a cerca y las resistencias de cerca a
    lejos, de forma que ``nearest_support`` y ``nearest_resistance`` son los
    niveles inmediatos contra los que se mide el recorrido disponible.
    """
    require_ohlc(frame)
    if lookback < 2 * pivot_window + 1:
        raise ValueError("lookback demasiado corto para el pivot_window pedido")

    recent = frame.iloc[-lookback:]
    price = float(reference_price if reference_price is not None else recent["close"].iloc[-1])

    highs = swing_highs(recent, pivot_window).dropna().tolist()
    lows = swing_lows(recent, pivot_window).dropna().tolist()
    return levels_from_pivots([*highs, *lows], reference_price=price, tolerance_pct=tolerance_pct)
