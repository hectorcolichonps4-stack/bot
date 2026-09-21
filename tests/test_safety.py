"""Tests del candado de spot: ni cortos ni apalancamiento.

El objetivo no es comprobar que una bandera vale ``false`` por defecto, sino
que operar en corto o con apalancamiento requiere desactivar el candado a
proposito y que, aun asi, no hay ninguna via implementada para hacerlo.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from bot.config.schema import Config
from bot.data.filters import SymbolFilters
from bot.execution import BacktestBroker, ExecutionError, PaperBroker, Side
from bot.risk import position_size
from bot.strategy import StrategyEngine

RAIZ = Path(__file__).resolve().parents[1]
NOW = pd.Timestamp("2024-06-05 12:00", tz="UTC")


def broker(balance: float = 10_000.0, *, spot_only: bool = True) -> BacktestBroker:
    b = BacktestBroker(initial_balance=balance, spot_only=spot_only)
    b.advance(NOW, {"BTCUSDT": {"close": 100.0}})
    return b


# ---------------------------------------------------------------- cortos

def test_spot_only_esta_activo_por_defecto():
    assert Config().risk.spot_only is True


def test_allow_short_esta_en_false_por_defecto():
    assert Config().gates.allow_short is False


def test_el_settings_de_produccion_no_permite_cortos():
    from bot.config import DEFAULT_CONFIG_PATH, load_config

    config = load_config(DEFAULT_CONFIG_PATH).config
    assert config.risk.spot_only is True
    assert config.gates.allow_short is False


def test_activar_cortos_con_el_candado_puesto_es_un_error():
    """No basta con poner allow_short: hay que desarmar el candado a proposito."""
    with pytest.raises(ValidationError, match="incompatible con risk.spot_only"):
        Config.model_validate({"gates": {"allow_short": True}})


def test_el_motor_solo_evalua_largos(loaded):
    assert StrategyEngine(loaded).sides() == (Side.LONG,)


def test_el_broker_simulado_rechaza_los_cortos():
    with pytest.raises(ExecutionError, match="cortos bloqueados"):
        broker().open_position("BTCUSDT", Side.SHORT, 1.0, stop_loss=105.0, take_profit=90.0)


def test_el_broker_de_papel_hereda_el_candado(loaded, tmp_path):
    from bot.data import DataFeed, ParquetCache, SyntheticProvider

    feed = DataFeed(SyntheticProvider(bars=300), loaded.config, cache=ParquetCache(tmp_path))
    papel = PaperBroker(feed, initial_balance=10_000.0, spot_only=True)
    papel.set_price("BTCUSDT", 100.0)
    with pytest.raises(ExecutionError, match="cortos bloqueados"):
        papel.open_position("BTCUSDT", Side.SHORT, 1.0, stop_loss=105.0, take_profit=90.0)


def test_el_broker_de_binance_rechaza_cortos_aunque_se_le_pidan():
    """Ultima linea de defensa: el codigo que habla con el exchange."""
    fuente = (RAIZ / "bot" / "execution" / "binance.py").read_text(encoding="utf-8")
    assert "el broker de spot no admite cortos" in fuente


def test_el_backtest_propaga_el_candado(loaded):
    from bot.backtesting import BacktestEngine
    from bot.data import SyntheticProvider

    proveedor = SyntheticProvider(bars=800, seed=3)
    frames = {"BTCUSDT": proveedor.fetch_klines("BTCUSDT", "15m")}
    resultado = BacktestEngine(loaded).run(frames, initial_balance=10_000.0)
    assert all(t.side is Side.LONG for t in resultado.trades)


# --------------------------------------------------------- apalancamiento

def test_una_posicion_mayor_que_el_capital_es_un_error_de_configuracion():
    with pytest.raises(ValidationError):
        Config.model_validate({"risk": {"max_position_pct": 150.0}})


def test_una_exposicion_mayor_que_el_capital_es_un_error_de_configuracion():
    with pytest.raises(ValidationError):
        Config.model_validate({"risk": {"max_exposure_pct": 200.0}})


def test_el_tamano_nunca_supera_el_efectivo_disponible():
    resultado = position_size(
        equity=10_000.0, entry_price=100.0, stop_loss=99.99, risk_per_trade_pct=10.0,
        max_position_pct=100.0, available_cash=1_000.0,
        filters=SymbolFilters.permissive("BTCUSDT"),
    )
    assert resultado.notional <= 1_000.0 + 1e-9


def test_el_efectivo_nunca_se_pone_en_negativo():
    """Sin margen no se puede gastar lo que no hay: el broker se niega."""
    b = broker(balance=1_000.0)
    with pytest.raises(ExecutionError, match="efectivo insuficiente"):
        b.open_position("BTCUSDT", Side.LONG, 100.0, stop_loss=95.0, take_profit=110.0)
    assert b.cash == 1_000.0


def test_la_exposicion_agregada_no_puede_pasar_del_capital():
    b = broker(balance=10_000.0)
    b.advance(NOW, {"BTCUSDT": {"close": 100.0}, "ETHUSDT": {"close": 100.0}})
    b.open_position("BTCUSDT", Side.LONG, 60.0, stop_loss=95.0, take_profit=110.0)
    with pytest.raises(ExecutionError):
        b.open_position("ETHUSDT", Side.LONG, 60.0, stop_loss=95.0, take_profit=110.0)


def test_no_hay_endpoints_de_margen_ni_futuros_en_el_codigo():
    """Aunque alguien quisiera apalancarse, no hay por donde: no esta escrito."""
    prohibidos = re.compile(r"/fapi|/dapi|/sapi/v1/margin|sideEffectType|isIsolated|leverage", re.IGNORECASE)
    for fichero in (RAIZ / "bot").rglob("*.py"):
        assert not prohibidos.search(fichero.read_text(encoding="utf-8")), f"referencia a margen en {fichero}"
