"""Construccion del cuadro de indicadores que consumen gates y puntuacion."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from bot.config.schema import IndicatorsConfig
from bot.indicators import (
    Levels,
    atr,
    atr_pct,
    ema,
    levels_from_pivots,
    macd,
    relative_volume,
    rsi,
    slope,
    swing_highs,
    swing_lows,
)


def build_features(frame: pd.DataFrame, config: IndicatorsConfig) -> pd.DataFrame:
    """Anade las columnas de indicadores al DataFrame de velas.

    Devuelve una copia: las velas originales siguen siendo la fuente de verdad
    y nunca se mutan.
    """
    out = frame.copy()
    close = out["close"]

    out["ema_fast"] = ema(close, config.ema_fast)
    out["ema_slow"] = ema(close, config.ema_slow)
    out["ema_trend"] = ema(close, config.ema_trend)
    out["ema_fast_slope"] = slope(out["ema_fast"], 5)

    out["rsi"] = rsi(close, config.rsi_period)

    macd_frame = macd(close, config.macd.fast, config.macd.slow, config.macd.signal)
    out["macd"] = macd_frame["macd"]
    out["macd_signal"] = macd_frame["signal"]
    out["macd_hist"] = macd_frame["hist"]
    out["macd_hist_prev"] = out["macd_hist"].shift(1)

    out["atr"] = atr(out, config.atr_period)
    out["atr_pct"] = atr_pct(out, config.atr_period)

    out["relative_volume"] = relative_volume(out["volume"], config.relative_volume_lookback)

    # Los pivotes se detectan una sola vez para todo el historico; recalcular
    # la ventana de soportes en cada barra era el cuello de botella del
    # backtest. Al consumirlos hay que descartar los ultimos sr_pivot_window
    # (ver ``snapshot``), porque todavia no estarian confirmados.
    out["swing_high"] = swing_highs(out, config.sr_pivot_window)
    out["swing_low"] = swing_lows(out, config.sr_pivot_window)
    return out


@dataclass(frozen=True)
class MarketSnapshot:
    """Estado del mercado en la ultima vela cerrada.

    Es lo unico que ven los gates y el scorer: un objeto plano y sin pandas,
    facil de construir en un test sin fabricar un DataFrame entero.
    """

    symbol: str
    timestamp: pd.Timestamp
    close: float
    high: float
    low: float
    volume: float
    quote_volume: float
    ema_fast: float
    ema_slow: float
    ema_trend: float
    ema_fast_slope: float
    rsi: float
    macd: float
    macd_signal: float
    macd_hist: float
    macd_hist_prev: float
    atr: float
    atr_pct: float
    relative_volume: float
    levels: Levels
    bars_available: int
    spread_pct: float = 0.0

    @property
    def trend_up(self) -> bool:
        return self.ema_fast > self.ema_slow > self.ema_trend

    @property
    def trend_down(self) -> bool:
        return self.ema_fast < self.ema_slow < self.ema_trend

    @property
    def nearest_support(self) -> float | None:
        return self.levels.nearest_support

    @property
    def nearest_resistance(self) -> float | None:
        return self.levels.nearest_resistance

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": str(self.timestamp),
            "close": self.close,
            "ema_fast": self.ema_fast,
            "ema_slow": self.ema_slow,
            "ema_trend": self.ema_trend,
            "rsi": self.rsi,
            "macd_hist": self.macd_hist,
            "atr_pct": self.atr_pct,
            "relative_volume": self.relative_volume,
            "support": self.nearest_support,
            "resistance": self.nearest_resistance,
            "spread_pct": self.spread_pct,
        }


class IncompleteSnapshot(ValueError):
    """No hay suficientes velas cerradas para calcular los indicadores."""


_REQUIRED = ("ema_trend", "rsi", "macd_hist", "atr", "relative_volume")

_SCALARS = (
    "close", "high", "low", "volume", "quote_volume",
    "ema_fast", "ema_slow", "ema_trend", "ema_fast_slope",
    "rsi", "macd", "macd_signal", "macd_hist", "macd_hist_prev",
    "atr", "atr_pct", "relative_volume",
)


class SnapshotBuilder:
    """Extrae ``MarketSnapshot`` de cualquier barra de un cuadro ya preparado.

    Convierte las columnas a arrays de numpy y aplana los pivotes una sola
    vez. Da igual pedir una barra o cien mil: en un backtest largo el acceso
    fila a fila de pandas costaba mas que todo el resto de la estrategia.
    """

    def __init__(self, symbol: str, features: pd.DataFrame, config: IndicatorsConfig) -> None:
        if features.empty:
            raise IncompleteSnapshot(f"{symbol}: sin velas")

        self.symbol = symbol.upper()
        self.config = config
        self.length = len(features)
        self.open_time = features["open_time"].to_numpy()
        self.columns = {
            name: features[name].to_numpy(dtype="float64", na_value=np.nan)
            for name in _SCALARS
            if name in features.columns
        }

        missing = [name for name in _REQUIRED if name not in self.columns]
        if missing:
            raise IncompleteSnapshot(f"{symbol}: faltan indicadores en el cuadro ({', '.join(missing)})")

        # Pivotes como pares (posicion, precio), ya ordenados por posicion:
        # localizar los de una ventana es entonces un par de bisecciones.
        pivot_positions: list[int] = []
        pivot_prices: list[float] = []
        for column in ("swing_high", "swing_low"):
            if column not in features.columns:
                continue
            values = features[column].to_numpy(dtype="float64", na_value=np.nan)
            for position in np.flatnonzero(~np.isnan(values)):
                pivot_positions.append(int(position))
                pivot_prices.append(float(values[position]))
        order = np.argsort(np.asarray(pivot_positions, dtype="int64")) if pivot_positions else np.array([], dtype="int64")
        self._pivot_positions = np.asarray(pivot_positions, dtype="int64")[order] if pivot_positions else np.array([], dtype="int64")
        self._pivot_prices = np.asarray(pivot_prices, dtype="float64")[order] if pivot_prices else np.array([], dtype="float64")

    def at(self, index: int = -1, *, spread_pct: float = 0.0) -> MarketSnapshot:
        """Estado del mercado en la barra ``index`` (negativo cuenta desde el final)."""
        position = index if index >= 0 else self.length + index
        if not 0 <= position < self.length:
            raise IncompleteSnapshot(f"{self.symbol}: barra {index} fuera de rango")

        values = {name: float(column[position]) for name, column in self.columns.items()}
        blank = [name for name in _REQUIRED if math.isnan(values[name])]
        if blank:
            raise IncompleteSnapshot(f"{self.symbol}: indicadores sin calentar ({', '.join(blank)})")

        close = values["close"]
        return MarketSnapshot(
            symbol=self.symbol,
            timestamp=pd.Timestamp(self.open_time[position]),
            close=close,
            high=values["high"],
            low=values["low"],
            volume=values["volume"],
            quote_volume=_or_zero(values.get("quote_volume")),
            ema_fast=values["ema_fast"],
            ema_slow=values["ema_slow"],
            ema_trend=values["ema_trend"],
            ema_fast_slope=_or_zero(values.get("ema_fast_slope")),
            rsi=values["rsi"],
            macd=values["macd"],
            macd_signal=values["macd_signal"],
            macd_hist=values["macd_hist"],
            macd_hist_prev=_or_zero(values.get("macd_hist_prev")),
            atr=values["atr"],
            atr_pct=values["atr_pct"],
            relative_volume=values["relative_volume"],
            levels=self.levels_at(position, close),
            bars_available=position + 1,
            spread_pct=spread_pct,
        )

    def levels_at(self, position: int, reference_price: float) -> Levels:
        """Soportes y resistencias vigentes en esa barra.

        Un pivote fractal necesita ``sr_pivot_window`` velas posteriores para
        confirmarse, asi que la ventana se corta ahi: incluir los ultimos
        seria usar informacion que en ese momento todavia no existia.
        """
        if self._pivot_positions.size == 0:
            return Levels((), ())

        last = position - self.config.sr_pivot_window
        first = last - self.config.sr_lookback
        if last < 0:
            return Levels((), ())

        lo = int(np.searchsorted(self._pivot_positions, max(first, 0), side="left"))
        hi = int(np.searchsorted(self._pivot_positions, last, side="right"))
        if hi <= lo:
            return Levels((), ())

        return levels_from_pivots(
            self._pivot_prices[lo:hi].tolist(),
            reference_price=reference_price,
            tolerance_pct=self.config.sr_tolerance_pct,
        )


def snapshot(
    symbol: str,
    features: pd.DataFrame,
    config: IndicatorsConfig,
    *,
    index: int = -1,
    spread_pct: float = 0.0,
) -> MarketSnapshot:
    """Atajo de un solo uso sobre ``SnapshotBuilder``.

    Para recorrer muchas barras conviene construir el builder una vez y
    llamar a ``at()``, que es justo lo que hace el motor de backtest.
    """
    return SnapshotBuilder(symbol, features, config).at(index, spread_pct=spread_pct)


def _or_zero(value: float | None) -> float:
    return 0.0 if value is None or math.isnan(value) else value
