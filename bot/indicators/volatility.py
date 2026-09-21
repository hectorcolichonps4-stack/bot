"""Rango verdadero y ATR."""

from __future__ import annotations

import pandas as pd

from bot.indicators._validate import require_ohlc


def true_range(frame: pd.DataFrame) -> pd.Series:
    """Rango verdadero: max(h-l, |h-c_prev|, |l-c_prev|).

    La primera barra no tiene cierre previo, asi que cae a ``high - low``.
    """
    require_ohlc(frame)
    prev_close = frame["close"].shift(1)
    ranges = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1, skipna=True).rename("true_range")


def atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR de Wilder (media exponencial del rango verdadero, alpha = 1/period)."""
    require_ohlc(frame)
    if period < 1:
        raise ValueError(f"el periodo debe ser >= 1, se recibio {period}")
    tr = true_range(frame)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean().rename(f"atr_{period}")


def atr_pct(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR expresado como porcentaje del cierre: comparable entre simbolos."""
    values = atr(frame, period) / frame["close"] * 100.0
    return values.rename(f"atr_pct_{period}")
