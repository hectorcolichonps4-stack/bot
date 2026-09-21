"""Proveedores de velas: REST de Binance, CSV local y generador sintetico."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from bot.data.filters import SymbolFilters
from bot.data.schema import (
    OHLCV_COLUMNS,
    DataError,
    empty_frame,
    normalize,
    timeframe_to_millis,
)

BINANCE_MAINNET = "https://api.binance.com"
BINANCE_TESTNET = "https://testnet.binance.vision"


class MarketDataProvider(ABC):
    """Fuente de velas. Todo lo demas del bot habla solo con esta interfaz."""

    @abstractmethod
    def fetch_klines(
        self,
        symbol: str,
        timeframe: str,
        *,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Devuelve velas normalizadas en el rango pedido (inclusive)."""

    def fetch_symbol_filters(self, symbol: str) -> SymbolFilters:
        """Filtros del simbolo; por defecto, permisivos (offline)."""
        return SymbolFilters.permissive(symbol)

    def fetch_ticker(self, symbol: str) -> Mapping[str, float]:
        """Ultimo precio y spread. Los proveedores offline lo derivan de la vela."""
        frame = self.fetch_klines(symbol, "1m", limit=1)
        if frame.empty:
            raise DataError(f"sin datos de ticker para {symbol}")
        close = float(frame["close"].iloc[-1])
        return {"bid": close, "ask": close, "last": close, "spread_pct": 0.0}


class BinanceRestProvider(MarketDataProvider):
    """Cliente REST publico de Binance (no requiere claves)."""

    def __init__(
        self,
        *,
        testnet: bool = True,
        base_url: str | None = None,
        timeout: float = 15.0,
        max_retries: int = 4,
        session_sleep=time.sleep,
    ) -> None:
        self.base_url = (base_url or (BINANCE_TESTNET if testnet else BINANCE_MAINNET)).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._sleep = session_sleep

    def _get(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "trading-bot/1.0"})
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                # 429/418 son rate limit: merece la pena esperar. El resto no.
                if exc.code not in (418, 429, 500, 502, 503, 504):
                    raise DataError(f"binance {exc.code} en {path}: {exc.read()[:200]!r}") from exc
                last_error = exc
            except urllib.error.URLError as exc:
                if _is_policy_denial(exc):
                    # Un proxy corporativo que deniega el destino no va a
                    # cambiar de idea: reintentar solo alarga el fallo.
                    raise DataError(
                        f"acceso a {self.base_url} denegado por la politica de red del entorno "
                        f"({exc.reason}). No es un fallo del bot: hay que permitir el host "
                        f"o alimentar las velas desde CSV con --csv-dir."
                    ) from exc
                last_error = exc
            except (TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
            self._sleep(2.0 ** attempt)
        raise DataError(f"binance no responde tras {self.max_retries} intentos en {path}: {last_error}")

    def fetch_klines(
        self,
        symbol: str,
        timeframe: str,
        *,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        step_ms = timeframe_to_millis(timeframe)
        start_ms = _to_millis(start)
        end_ms = _to_millis(end)

        chunks: list[pd.DataFrame] = []
        cursor = start_ms
        while True:
            params: dict[str, Any] = {
                "symbol": symbol.upper(),
                "interval": timeframe,
                "limit": min(limit, 1000),
            }
            if cursor is not None:
                params["startTime"] = cursor
            if end_ms is not None:
                params["endTime"] = end_ms

            payload = self._get("/api/v3/klines", params)
            if not payload:
                break

            chunks.append(_klines_to_frame(payload))
            if cursor is None or len(payload) < params["limit"]:
                break

            cursor = int(payload[-1][0]) + step_ms
            if end_ms is not None and cursor > end_ms:
                break

        if not chunks:
            return empty_frame()
        return normalize(pd.concat(chunks, ignore_index=True))

    def fetch_symbol_filters(self, symbol: str) -> SymbolFilters:
        payload = self._get("/api/v3/exchangeInfo", {"symbol": symbol.upper()})
        symbols = payload.get("symbols") or []
        if not symbols:
            raise DataError(f"simbolo desconocido en el exchange: {symbol}")
        return SymbolFilters.from_binance(symbols[0])

    def fetch_ticker(self, symbol: str) -> Mapping[str, float]:
        book = self._get("/api/v3/ticker/bookTicker", {"symbol": symbol.upper()})
        bid = float(book["bidPrice"])
        ask = float(book["askPrice"])
        mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else max(bid, ask)
        spread_pct = ((ask - bid) / mid * 100.0) if mid > 0 else 0.0
        return {"bid": bid, "ask": ask, "last": mid, "spread_pct": spread_pct}


class CsvProvider(MarketDataProvider):
    """Lee velas de ficheros ``<root>/<SYMBOL>_<timeframe>.csv``.

    Util para reproducir un backtest con datos auditados sin depender de red.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def fetch_klines(
        self,
        symbol: str,
        timeframe: str,
        *,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        path = self.root / f"{symbol.upper()}_{timeframe}.csv"
        if not path.exists():
            raise DataError(f"no existe el CSV de velas: {path}")
        frame = normalize(pd.read_csv(path))
        return _slice(frame, start, end, limit)


class SyntheticProvider(MarketDataProvider):
    """Genera velas deterministas. Sirve para demos, tests y humo end-to-end.

    No pretende imitar un mercado real: solo produce series con tendencia y
    ruido reproducibles a partir de una semilla.
    """

    def __init__(
        self,
        *, 
        seed: int = 7,
        start_price: float = 100.0,
        drift: float = 0.0002,
        volatility: float = 0.008,
        annual_drift: float | None = None,
        annual_volatility: float | None = None,
        bars: int = 2000,
        base_volume: float = 50_000.0,
        anchor: pd.Timestamp | None = None,
    ) -> None:
        self.seed = seed
        self.start_price = start_price
        self.drift = drift
        self.volatility = volatility
        self.annual_drift = annual_drift
        self.annual_volatility = annual_volatility
        self.bars = bars
        self.base_volume = base_volume
        self.anchor = anchor or pd.Timestamp.now(tz="UTC").floor("min")

    def _per_bar(self, step: pd.Timedelta) -> tuple[float, float]:
        """Convierte drift y volatilidad anuales a la escala de una vela.

        Sin esta conversion, un drift por vela razonable a 15m se vuelve
        absurdo a 1h sobre cinco anos: compuesto sobre 50.000 velas convierte
        un precio de 100 en varios millones, y cualquier comparacion contra
        comprar y mantener deja de significar nada.
        """
        if self.annual_drift is None and self.annual_volatility is None:
            return self.drift, self.volatility

        bars_per_year = pd.Timedelta(days=365) / step
        drift = (self.annual_drift if self.annual_drift is not None else 0.0) / bars_per_year
        volatility = (
            (self.annual_volatility if self.annual_volatility is not None else 0.5)
            / np.sqrt(bars_per_year)
        )
        return float(drift), float(volatility)

    def fetch_klines(
        self,
        symbol: str,
        timeframe: str,
        *,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        step = pd.Timedelta(milliseconds=timeframe_to_millis(timeframe))
        count = self.bars
        # La semilla depende del simbolo: cada par tiene su propia serie.
        rng = np.random.default_rng(self.seed + (abs(hash(symbol.upper())) % 10_000))

        drift, volatility = self._per_bar(step)
        returns = rng.normal(drift, volatility, count)
        close = self.start_price * np.exp(np.cumsum(returns))
        spread = np.abs(rng.normal(0, volatility / 2, count)) * close
        open_ = np.concatenate([[self.start_price], close[:-1]])
        high = np.maximum(open_, close) + spread
        low = np.minimum(open_, close) - spread
        # Volumen centrado en ``base_volume`` para que el producto por el
        # precio quede en el orden de magnitud de un par liquido real.
        volume = self.base_volume * rng.lognormal(mean=0.0, sigma=0.5, size=count)

        open_time = pd.date_range(end=self.anchor, periods=count, freq=step, tz="UTC")
        frame = pd.DataFrame(
            {
                "open_time": open_time,
                "open": open_,
                "high": high,
                "low": np.maximum(low, 1e-8),
                "close": close,
                "volume": volume,
                "close_time": open_time + step - pd.Timedelta(milliseconds=1),
                "quote_volume": volume * close,
                "trades": rng.integers(50, 500, count),
            }
        )
        return _slice(normalize(frame), start, end, limit)


def _slice(
    frame: pd.DataFrame,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    limit: int,
) -> pd.DataFrame:
    out = frame
    if start is not None:
        out = out[out["open_time"] >= _as_utc(start)]
    if end is not None:
        out = out[out["open_time"] <= _as_utc(end)]
    if limit and len(out) > limit and start is None:
        out = out.iloc[-limit:]
    return out.reset_index(drop=True)


_POLICY_MARKERS = ("tunnel connection failed", "403", "407", "proxy")


def _is_policy_denial(exc: urllib.error.URLError) -> bool:
    """Distingue "la red va mal" de "este destino esta prohibido"."""
    return any(marker in str(exc.reason).lower() for marker in _POLICY_MARKERS)


def _as_utc(value: pd.Timestamp | str) -> pd.Timestamp:
    """Interpreta como UTC cualquier marca temporal sin zona horaria."""
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _klines_to_frame(payload: Sequence[Sequence[Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in payload:
        rows.append(
            {
                "open_time": pd.to_datetime(int(row[0]), unit="ms", utc=True),
                "open": row[1],
                "high": row[2],
                "low": row[3],
                "close": row[4],
                "volume": row[5],
                "close_time": pd.to_datetime(int(row[6]), unit="ms", utc=True),
                "quote_volume": row[7],
                "trades": row[8],
            }
        )
    return pd.DataFrame(rows, columns=list(OHLCV_COLUMNS))


def _to_millis(value: pd.Timestamp | None) -> int | None:
    if value is None:
        return None
    return int(_as_utc(value).timestamp() * 1000)
