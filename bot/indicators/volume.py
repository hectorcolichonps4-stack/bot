"""Indicadores de volumen."""

from __future__ import annotations

import pandas as pd

from bot.indicators._validate import require_numeric_series


def relative_volume(volume: pd.Series, lookback: int = 20) -> pd.Series:
    """Volumen actual dividido por la media de las ``lookback`` barras previas.

    La media excluye la barra en curso (``shift(1)``): incluirla diluiria
    justo el pico que se quiere detectar. 1.0 es volumen normal, 2.0 el doble.
    """
    require_numeric_series(volume, lookback)
    baseline = volume.shift(1).rolling(window=lookback, min_periods=lookback).mean()
    out = volume / baseline
    return out.replace([float("inf"), float("-inf")], float("nan")).rename("relative_volume")


def volume_trend(volume: pd.Series, lookback: int = 20) -> pd.Series:
    """Pendiente relativa de la media de volumen: acumulacion vs distribucion."""
    require_numeric_series(volume, lookback)
    avg = volume.rolling(window=lookback, min_periods=lookback).mean()
    return ((avg - avg.shift(lookback)) / avg.shift(lookback)).rename("volume_trend")
