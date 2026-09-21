"""Formato canonico de las velas que circula por todo el bot."""

from __future__ import annotations

import pandas as pd

OHLCV_COLUMNS: tuple[str, ...] = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
)

NUMERIC_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume", "quote_volume", "trades")


class DataError(RuntimeError):
    """Datos ausentes, incoherentes o corruptos."""


def empty_frame() -> pd.DataFrame:
    """DataFrame vacio pero con el esquema correcto (evita ramas especiales)."""
    frame = pd.DataFrame({column: pd.Series(dtype="float64") for column in NUMERIC_COLUMNS})
    frame.insert(0, "open_time", pd.Series(dtype="datetime64[ns, UTC]"))
    frame["close_time"] = pd.Series(dtype="datetime64[ns, UTC]")
    return frame[list(OHLCV_COLUMNS)]


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Ordena, deduplica y tipa un DataFrame de velas.

    Es el unico punto donde se decide el contrato: indice temporal UTC
    ascendente, sin duplicados y con las columnas numericas en float.
    """
    if frame.empty:
        return empty_frame()

    missing = [column for column in OHLCV_COLUMNS if column not in frame.columns]
    if missing:
        raise DataError(f"faltan columnas en las velas: {missing}")

    out = frame.loc[:, list(OHLCV_COLUMNS)].copy()
    for column in ("open_time", "close_time"):
        out[column] = pd.to_datetime(out[column], utc=True)
    for column in NUMERIC_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("float64")

    out = out.dropna(subset=["open_time", "open", "high", "low", "close"])
    out = out.drop_duplicates(subset="open_time", keep="last")
    out = out.sort_values("open_time").reset_index(drop=True)
    return out


def validate(frame: pd.DataFrame, *, timeframe: str | None = None) -> pd.DataFrame:
    """Comprueba invariantes de mercado y devuelve el mismo frame.

    Un high por debajo del low, o un cierre fuera del rango, significa datos
    corruptos: es preferible abortar que operar sobre ellos.
    """
    if frame.empty:
        return frame

    bad_range = frame["high"] < frame["low"]
    if bad_range.any():
        raise DataError(f"{int(bad_range.sum())} velas con high < low")

    outside = (
        (frame["close"] > frame["high"])
        | (frame["close"] < frame["low"])
        | (frame["open"] > frame["high"])
        | (frame["open"] < frame["low"])
    )
    if outside.any():
        raise DataError(f"{int(outside.sum())} velas con open/close fuera del rango high-low")

    if (frame["volume"] < 0).any():
        raise DataError("volumen negativo en las velas")

    if timeframe is not None and len(frame) > 2:
        expected = timeframe_to_timedelta(timeframe)
        gaps = frame["open_time"].diff().dropna()
        irregular = gaps[gaps != expected]
        if len(irregular) > 0:
            # No es fatal: los exchanges tienen paradas. Se anota para el log.
            frame.attrs["gaps"] = int(len(irregular))
    return frame


_TIMEFRAME_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720, "1d": 1440,
}


def timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    if timeframe not in _TIMEFRAME_MINUTES:
        raise DataError(f"timeframe no soportado: {timeframe}")
    return pd.Timedelta(minutes=_TIMEFRAME_MINUTES[timeframe])


def timeframe_to_millis(timeframe: str) -> int:
    return int(timeframe_to_timedelta(timeframe).total_seconds() * 1000)
