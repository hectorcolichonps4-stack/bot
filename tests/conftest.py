"""Utilidades compartidas por los tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot.config.loader import LoadedConfig, compute_config_hash
from bot.config.schema import Config
from bot.execution.base import ClosedTrade, Side
from bot.indicators import Levels
from bot.strategy.features import MarketSnapshot


@pytest.fixture
def config() -> Config:
    return Config()


@pytest.fixture
def loaded(config: Config) -> LoadedConfig:
    return LoadedConfig(config=config, config_hash=compute_config_hash(config), source_path=None, raw={})


@pytest.fixture
def candles() -> pd.DataFrame:
    """Velas deterministas con tendencia alcista suave."""
    rng = np.random.default_rng(42)
    count = 500
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0008, 0.01, count)))
    spread = np.abs(rng.normal(0, 0.004, count)) * close
    open_ = np.concatenate([[100.0], close[:-1]])
    open_time = pd.date_range("2024-01-01", periods=count, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "open_time": open_time,
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
            "volume": rng.lognormal(10.0, 0.4, count),
            "close_time": open_time + pd.Timedelta(minutes=15),
            "quote_volume": rng.lognormal(10.0, 0.4, count) * close,
            "trades": rng.integers(100, 900, count),
        }
    )


def make_snapshot(**overrides) -> MarketSnapshot:
    """Snapshot valido por defecto; cada test cambia solo lo que le interesa."""
    defaults = dict(
        symbol="BTCUSDT",
        timestamp=pd.Timestamp("2024-06-01 12:00", tz="UTC"),
        close=100.0,
        high=101.0,
        low=99.0,
        volume=1_000.0,
        quote_volume=10_000_000.0,
        ema_fast=99.0,
        ema_slow=97.0,
        ema_trend=95.0,
        ema_fast_slope=0.01,
        rsi=58.0,
        macd=0.5,
        macd_signal=0.3,
        macd_hist=0.2,
        macd_hist_prev=0.1,
        atr=1.5,
        atr_pct=1.5,
        relative_volume=1.8,
        levels=Levels(supports=(96.0,), resistances=(108.0,)),
        bars_available=400,
        spread_pct=0.02,
    )
    defaults.update(overrides)
    return MarketSnapshot(**defaults)


def make_trade(
    *,
    pnl: float,
    symbol: str = "BTCUSDT",
    closed_at: pd.Timestamp | str = "2024-06-01 12:00",
    opened_at: pd.Timestamp | str | None = None,
    r_multiple: float | None = None,
) -> ClosedTrade:
    closed = pd.Timestamp(closed_at, tz="UTC") if pd.Timestamp(closed_at).tzinfo is None else pd.Timestamp(closed_at)
    opened = pd.Timestamp(opened_at, tz="UTC") if opened_at else closed - pd.Timedelta(hours=1)
    return ClosedTrade(
        symbol=symbol,
        side=Side.LONG,
        quantity=1.0,
        entry_price=100.0,
        exit_price=100.0 + pnl,
        opened_at=opened,
        closed_at=closed,
        pnl=pnl,
        fees=0.1,
        r_multiple=r_multiple if r_multiple is not None else pnl / 5.0,
        exit_reason="take_profit" if pnl > 0 else "stop_loss",
    )
