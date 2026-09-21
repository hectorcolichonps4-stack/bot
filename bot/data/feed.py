"""Orquestacion de datos: cache primero, exchange solo para lo que falta."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable

import pandas as pd

from bot.config.schema import Config
from bot.data.cache import ParquetCache
from bot.data.filters import SymbolFilters
from bot.data.providers import MarketDataProvider
from bot.data.schema import DataError, normalize, timeframe_to_timedelta, validate

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeedStats:
    symbol: str
    bars: int
    from_cache: int
    downloaded: int
    first: pd.Timestamp | None
    last: pd.Timestamp | None


class DataFeed:
    """Entrega velas listas para los indicadores.

    Politica: lo que ya esta en cache no se vuelve a pedir; del exchange solo
    se descarga el tramo que falta al final (o el rango completo si la cache
    esta vacia). La ultima vela en curso se descarta salvo que se pida
    ``include_open``, porque sus valores todavia cambian.
    """

    def __init__(
        self,
        provider: MarketDataProvider,
        config: Config,
        *,
        cache: ParquetCache | None = None,
    ) -> None:
        self.provider = provider
        self.config = config
        self.cache = cache or ParquetCache(config.data.cache_dir)
        self._filters: Dict[str, SymbolFilters] = {}
        self.stats: Dict[str, FeedStats] = {}

    # ------------------------------------------------------------------ velas

    def history(
        self,
        symbol: str,
        *,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        timeframe: str | None = None,
        refresh: bool = True,
    ) -> pd.DataFrame:
        """Historico completo del simbolo, usando la cache como base."""
        timeframe = timeframe or self.config.data.timeframe
        cached = self.cache.load(symbol, timeframe)
        cached_bars = len(cached)
        downloaded = 0

        if refresh:
            fetch_from = start
            if not cached.empty:
                # Se resolicita la ultima vela cacheada por si estaba abierta.
                fetch_from = cached["open_time"].iloc[-1]
                if start is not None and start < fetch_from:
                    fetch_from = start

            try:
                fresh = self.provider.fetch_klines(
                    symbol,
                    timeframe,
                    start=fetch_from,
                    end=end,
                    limit=self.config.data.max_bars_per_request,
                )
            except DataError:
                if cached.empty:
                    raise
                log.warning("descarga fallida para %s; se usa la cache", symbol, exc_info=True)
                fresh = cached.iloc[0:0]

            downloaded = len(fresh)
            if downloaded:
                cached = self.cache.merge(symbol, timeframe, fresh)

        frame = _clip(cached, start, end)
        frame = validate(normalize(frame), timeframe=timeframe)

        self.stats[symbol] = FeedStats(
            symbol=symbol,
            bars=len(frame),
            from_cache=cached_bars,
            downloaded=downloaded,
            first=frame["open_time"].iloc[0] if len(frame) else None,
            last=frame["open_time"].iloc[-1] if len(frame) else None,
        )
        return frame

    def latest(self, symbol: str, *, bars: int | None = None, include_open: bool = False) -> pd.DataFrame:
        """Ultimas ``bars`` velas cerradas, listas para evaluar una senal."""
        bars = bars or self.config.data.warmup_bars
        frame = self.history(symbol, refresh=True)
        if not include_open:
            frame = drop_unclosed(frame, self.config.data.timeframe)
        if len(frame) < bars:
            return frame.reset_index(drop=True)
        return frame.iloc[-bars:].reset_index(drop=True)

    def bulk_history(
        self,
        symbols: Iterable[str] | None = None,
        *,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        refresh: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Historico de varios simbolos; un fallo aislado no tumba el resto."""
        result: Dict[str, pd.DataFrame] = {}
        for symbol in symbols or self.config.data.symbols:
            try:
                result[symbol] = self.history(symbol, start=start, end=end, refresh=refresh)
            except DataError:
                log.error("sin datos utilizables para %s", symbol, exc_info=True)
        if not result:
            raise DataError("ningun simbolo devolvio datos")
        return result

    # ---------------------------------------------------------------- mercado

    def filters(self, symbol: str) -> SymbolFilters:
        """Filtros del simbolo, cacheados en memoria durante la sesion."""
        key = symbol.upper()
        if key not in self._filters:
            try:
                self._filters[key] = self.provider.fetch_symbol_filters(key)
            except DataError:
                log.warning("sin filtros de exchange para %s; se usan permisivos", key)
                self._filters[key] = SymbolFilters.permissive(key)
        return self._filters[key]

    def ticker(self, symbol: str) -> dict[str, float]:
        return dict(self.provider.fetch_ticker(symbol))


def drop_unclosed(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Elimina la ultima vela si su intervalo todavia no ha terminado."""
    if frame.empty:
        return frame
    step = timeframe_to_timedelta(timeframe)
    now = pd.Timestamp.now(tz="UTC")
    if frame["open_time"].iloc[-1] + step > now:
        return frame.iloc[:-1]
    return frame


def _clip(frame: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.DataFrame:
    out = frame
    if start is not None:
        out = out[out["open_time"] >= _utc(start)]
    if end is not None:
        out = out[out["open_time"] <= _utc(end)]
    return out.reset_index(drop=True)


def _utc(value: pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
