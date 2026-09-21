"""Osciladores de momento: RSI y MACD."""

from __future__ import annotations

import pandas as pd

from bot.indicators._validate import require_numeric_series


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI de Wilder (suavizado exponencial con alpha = 1/period).

    Devuelve valores en [0, 100]. Un tramo sin perdidas da exactamente 100 y
    uno sin ganancias exactamente 0, que es el comportamiento de referencia.
    """
    require_numeric_series(series, period)
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    out = 100.0 - (100.0 / (1.0 + rs))
    # avg_loss == 0 deja rs en inf -> 100; ambos a 0 (mercado plano) -> 50.
    out = out.where(avg_loss != 0, 100.0)
    out = out.where(~((avg_loss == 0) & (avg_gain == 0)), 50.0)
    out[avg_gain.isna() | avg_loss.isna()] = float("nan")
    return out.rename(f"rsi_{period}")


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD clasico. Devuelve las columnas ``macd``, ``signal`` e ``hist``."""
    require_numeric_series(series, fast)
    if fast >= slow:
        raise ValueError(f"fast ({fast}) debe ser menor que slow ({slow})")
    if signal < 1:
        raise ValueError(f"signal debe ser >= 1, se recibio {signal}")

    ema_fast = series.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = series.ewm(span=slow, adjust=False, min_periods=slow).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})
