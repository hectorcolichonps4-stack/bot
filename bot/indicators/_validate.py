"""Validaciones compartidas por los indicadores."""

from __future__ import annotations

import pandas as pd


def require_numeric_series(series: pd.Series, period: int | None = None) -> None:
    """Falla rapido ante entradas que producirian NaN silenciosos."""
    if not isinstance(series, pd.Series):
        raise TypeError(f"se esperaba pandas.Series, se recibio {type(series).__name__}")
    if period is not None and period < 1:
        raise ValueError(f"el periodo debe ser >= 1, se recibio {period}")
    # ``is_numeric_dtype`` cubre tanto el dtype ``object`` de pandas 2 como el
    # ``StringDtype`` que infiere pandas 3 para series de texto.
    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError(f"la serie debe ser numerica, se recibio dtype {series.dtype}")


def require_ohlc(frame: pd.DataFrame, columns: tuple[str, ...] = ("high", "low", "close")) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"se esperaba pandas.DataFrame, se recibio {type(frame).__name__}")
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"faltan columnas obligatorias: {missing}")
