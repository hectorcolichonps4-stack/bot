"""Capa de datos: descarga, cache en parquet y filtros del exchange."""

from bot.data.cache import ParquetCache
from bot.data.feed import DataFeed, FeedStats, drop_unclosed
from bot.data.filters import FilterError, SymbolFilters
from bot.data.providers import (
    BinanceRestProvider,
    CsvProvider,
    MarketDataProvider,
    SyntheticProvider,
)
from bot.data.schema import (
    NUMERIC_COLUMNS,
    OHLCV_COLUMNS,
    DataError,
    empty_frame,
    normalize,
    timeframe_to_millis,
    timeframe_to_timedelta,
    validate,
)

__all__ = [
    "BinanceRestProvider",
    "CsvProvider",
    "DataError",
    "DataFeed",
    "FeedStats",
    "FilterError",
    "MarketDataProvider",
    "NUMERIC_COLUMNS",
    "OHLCV_COLUMNS",
    "ParquetCache",
    "SymbolFilters",
    "SyntheticProvider",
    "drop_unclosed",
    "empty_frame",
    "normalize",
    "timeframe_to_millis",
    "timeframe_to_timedelta",
    "validate",
]
