"""Medias moviles y derivados de tendencia. Funciones puras sobre Series."""

from __future__ import annotations

import pandas as pd

from bot.indicators._validate import require_numeric_series


def sma(series: pd.Series, period: int) -> pd.Series:
    """Media movil simple."""
    require_numeric_series(series, period)
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Media movil exponencial con factor 2/(period+1).

    ``adjust=False`` reproduce la formula recursiva que usan los graficos de
    trading; los primeros ``period-1`` valores quedan como NaN para no emitir
    lecturas basadas en una ventana incompleta.
    """
    require_numeric_series(series, period)
    out = series.ewm(span=period, adjust=False, min_periods=period).mean()
    return out.rename(f"ema_{period}")


def slope(series: pd.Series, period: int = 5) -> pd.Series:
    """Pendiente normalizada: variacion relativa en ``period`` barras."""
    require_numeric_series(series, period)
    shifted = series.shift(period)
    return ((series - shifted) / shifted.abs()).replace([float("inf"), float("-inf")], pd.NA).astype(float)
