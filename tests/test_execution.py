"""Tests de ejecucion: contabilidad simulada, filtros y gestion de salidas."""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from bot.data.filters import FilterError, SymbolFilters
from bot.execution import (
    BacktestBroker,
    ExecutionError,
    Order,
    OrderType,
    Position,
    Side,
    SimulatedBroker,
)

NOW = pd.Timestamp("2024-06-05 12:00", tz="UTC")


def broker(balance: float = 10_000.0, *, fee_bps: float = 0.0, slippage_bps: float = 0.0) -> BacktestBroker:
    b = BacktestBroker(initial_balance=balance, fee_bps=fee_bps, slippage_bps=slippage_bps)
    b.advance(NOW, {"BTCUSDT": {"close": 100.0}})
    return b


# ------------------------------------------------------------------ filtros

def test_el_precio_se_ajusta_al_tick():
    filtros = SymbolFilters("BTCUSDT", tick_size=Decimal("0.5"))
    assert filtros.round_price(100.3) == 100.5
    assert filtros.round_price(100.2) == 100.0


def test_la_cantidad_siempre_se_trunca():
    """Redondear al alza dejaria la posicion por encima del riesgo aprobado."""
    filtros = SymbolFilters("BTCUSDT", step_size=Decimal("0.1"))
    assert filtros.round_qty(1.99) == 1.9


def test_se_rechaza_un_notional_insuficiente():
    filtros = SymbolFilters("BTCUSDT", min_notional=Decimal("100"))
    with pytest.raises(FilterError, match="notional"):
        filtros.validate_order(price=10.0, qty=1.0)


def test_se_rechaza_una_cantidad_por_debajo_del_minimo():
    filtros = SymbolFilters("BTCUSDT", min_qty=Decimal("1"), min_notional=Decimal("0"))
    with pytest.raises(FilterError, match="minQty"):
        filtros.validate_order(price=100.0, qty=0.5)


def test_los_filtros_se_leen_de_exchange_info():
    filtros = SymbolFilters.from_binance({
        "symbol": "btcusdt",
        "baseAsset": "BTC",
        "quoteAsset": "USDT",
        "filters": [
            {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
            {"filterType": "LOT_SIZE", "stepSize": "0.00001", "minQty": "0.00001", "maxQty": "9000"},
            {"filterType": "NOTIONAL", "minNotional": "5.0"},
        ],
    })
    assert filtros.symbol == "BTCUSDT"
    assert filtros.tick_size == Decimal("0.10")
    assert filtros.min_notional == Decimal("5.0")


# ------------------------------------------------- contabilidad del broker

def test_abrir_posicion_descuenta_el_efectivo():
    b = broker()
    b.open_position("BTCUSDT", Side.LONG, 10.0, stop_loss=95.0, take_profit=110.0)
    assert b.cash == pytest.approx(9_000.0)
    assert b.equity() == pytest.approx(10_000.0)


def test_el_equity_refleja_el_beneficio_latente():
    b = broker()
    b.open_position("BTCUSDT", Side.LONG, 10.0, stop_loss=95.0, take_profit=110.0)
    b.advance(NOW, {"BTCUSDT": {"close": 110.0}})
    assert b.equity() == pytest.approx(10_100.0)


def test_cerrar_devuelve_el_capital_mas_el_resultado():
    b = broker()
    b.open_position("BTCUSDT", Side.LONG, 10.0, stop_loss=95.0, take_profit=110.0)
    b.advance(NOW, {"BTCUSDT": {"close": 110.0}})
    b.close_position("BTCUSDT", reason="manual")
    assert b.cash == pytest.approx(10_100.0)
    assert b.closed_trades[-1].pnl == pytest.approx(100.0)


def test_las_comisiones_reducen_el_resultado():
    b = broker(fee_bps=10.0)  # 0,1% por lado
    b.open_position("BTCUSDT", Side.LONG, 10.0, stop_loss=95.0, take_profit=110.0)
    b.advance(NOW, {"BTCUSDT": {"close": 100.0}})
    b.close_position("BTCUSDT")
    assert b.closed_trades[-1].pnl == pytest.approx(-2.0)  # 1 + 1 de comision


def test_el_deslizamiento_siempre_va_en_contra():
    b = broker(slippage_bps=100.0)  # 1%
    b.open_position("BTCUSDT", Side.LONG, 10.0, stop_loss=95.0, take_profit=110.0)
    assert b.position_for("BTCUSDT").entry_price == pytest.approx(101.0)


def test_no_se_puede_abrir_sin_efectivo_suficiente():
    b = broker(balance=100.0)
    with pytest.raises(ExecutionError, match="efectivo insuficiente"):
        b.open_position("BTCUSDT", Side.LONG, 10.0, stop_loss=95.0, take_profit=110.0)


def test_no_se_duplica_posicion_en_el_mismo_simbolo():
    b = broker()
    b.open_position("BTCUSDT", Side.LONG, 5.0, stop_loss=95.0, take_profit=110.0)
    with pytest.raises(ExecutionError, match="ya hay una posicion"):
        b.open_position("BTCUSDT", Side.LONG, 5.0, stop_loss=95.0, take_profit=110.0)


def test_cerrar_un_simbolo_sin_posicion_no_hace_nada():
    assert broker().close_position("ETHUSDT") is None


# ---------------------------------------------------------- stops y objetivos

def test_el_stop_se_ejecuta_al_tocar_el_rango():
    b = broker()
    b.open_position("BTCUSDT", Side.LONG, 10.0, stop_loss=95.0, take_profit=110.0)
    cerradas = b.check_exits({"BTCUSDT": {"high": 99.0, "low": 94.0}})
    assert len(cerradas) == 1 and cerradas[0].exit_reason == "stop_loss"
    assert cerradas[0].exit_price == pytest.approx(95.0)


def test_el_objetivo_se_ejecuta_al_tocar_el_rango():
    b = broker()
    b.open_position("BTCUSDT", Side.LONG, 10.0, stop_loss=95.0, take_profit=110.0)
    cerradas = b.check_exits({"BTCUSDT": {"high": 111.0, "low": 105.0}})
    assert cerradas[0].exit_reason == "take_profit"


def test_si_la_vela_toca_ambos_se_asume_el_stop():
    """Sin datos de tick no se sabe el orden: suponer lo peor evita un backtest optimista."""
    b = broker()
    b.open_position("BTCUSDT", Side.LONG, 10.0, stop_loss=95.0, take_profit=110.0)
    cerradas = b.check_exits({"BTCUSDT": {"high": 115.0, "low": 90.0}})
    assert cerradas[0].exit_reason == "stop_loss"


def test_una_vela_dentro_del_rango_no_cierra_nada():
    b = broker()
    b.open_position("BTCUSDT", Side.LONG, 10.0, stop_loss=95.0, take_profit=110.0)
    assert b.check_exits({"BTCUSDT": {"high": 105.0, "low": 98.0}}) == []


def test_close_all_cierra_todas_las_posiciones():
    b = broker()
    b.advance(NOW, {"BTCUSDT": {"close": 100.0}, "ETHUSDT": {"close": 50.0}})
    b.open_position("BTCUSDT", Side.LONG, 5.0, stop_loss=95.0, take_profit=110.0)
    b.open_position("ETHUSDT", Side.LONG, 10.0, stop_loss=45.0, take_profit=60.0)
    assert len(b.close_all()) == 2
    assert b.positions() == ()


# ------------------------------------------------------------- posiciones

def test_el_multiplo_r_mide_el_resultado_en_unidades_de_riesgo():
    posicion = Position("BTCUSDT", Side.LONG, 1.0, 100.0, 95.0, 110.0, NOW)
    assert posicion.r_multiple(105.0) == pytest.approx(1.0)
    assert posicion.r_multiple(95.0) == pytest.approx(-1.0)


def test_un_corto_gana_cuando_el_precio_baja():
    posicion = Position("BTCUSDT", Side.SHORT, 1.0, 100.0, 105.0, 90.0, NOW)
    assert posicion.unrealized_pnl(95.0) == pytest.approx(5.0)
    assert posicion.hit_target(low=89.0, high=96.0)
    assert posicion.hit_stop(low=100.0, high=106.0)


def test_una_orden_sin_cantidad_es_invalida():
    with pytest.raises(ValueError):
        Order(symbol="BTCUSDT", side=Side.LONG, quantity=0.0)


def test_una_orden_limit_exige_precio():
    with pytest.raises(ValueError, match="precio"):
        Order(symbol="BTCUSDT", side=Side.LONG, quantity=1.0, order_type=OrderType.LIMIT)


def test_el_saldo_inicial_debe_ser_positivo():
    with pytest.raises(ValueError):
        SimulatedBroker(initial_balance=0.0)


def test_la_operacion_cerrada_conserva_su_stop_y_su_objetivo():
    """El registro debe poder reconstruir la gestion real, no aproximarla."""
    b = broker()
    b.open_position("BTCUSDT", Side.LONG, 10.0, stop_loss=95.0, take_profit=110.0)
    b.check_exits({"BTCUSDT": {"high": 111.0, "low": 105.0}})
    trade = b.closed_trades[-1]
    assert trade.stop_loss == 95.0 and trade.take_profit == 110.0
