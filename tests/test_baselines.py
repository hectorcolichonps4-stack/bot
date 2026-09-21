"""Tests de las estrategias de referencia y de los informes por periodo."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bot.backtesting.baselines import (
    make_ema_rule,
    make_rsi_macd_rule,
    run_baseline,
    rule_buy_and_hold,
    run_rule,
    standard_baselines,
)
from bot.backtesting.reporting import (
    comparison_table,
    equity_to_frame,
    monthly_activity_summary,
    split_by_period,
    trades_per_month,
)
from bot.data import SyntheticProvider

from tests.conftest import make_trade


@pytest.fixture
def frames():
    proveedor = SyntheticProvider(bars=2000, seed=9, drift=0.0004, anchor=pd.Timestamp("2024-12-31", tz="UTC"))
    return {s: proveedor.fetch_klines(s, "1h") for s in ("BTCUSDT", "ETHUSDT")}


def rampa(precios: list[float]) -> pd.DataFrame:
    tiempos = pd.date_range("2024-01-01", periods=len(precios), freq="1h", tz="UTC")
    serie = pd.Series(precios, dtype="float64")
    return pd.DataFrame(
        {
            "open_time": tiempos, "open": serie, "high": serie * 1.001, "low": serie * 0.999,
            "close": serie, "volume": 1000.0, "close_time": tiempos, "quote_volume": serie * 1000.0,
            "trades": 100.0,
        }
    )


# ---------------------------------------------------------------- reglas

def test_comprar_y_mantener_sigue_el_precio_sin_costes():
    frame = rampa([100.0, 110.0, 120.0, 130.0])
    equity, trades = run_rule(
        frame, rule_buy_and_hold, symbol="X", initial_balance=1_000.0, fee_bps=0.0, slippage_bps=0.0
    )
    # Entra en la apertura de la segunda vela (110) y acaba en 130.
    assert equity.iloc[-1] == pytest.approx(1_000.0 * 130.0 / 110.0)
    assert len(trades) == 1


def test_la_referencia_tambien_ejecuta_con_un_retardo():
    """Sin el retardo la referencia compraria al precio que genero su senal."""
    frame = rampa([100.0, 200.0, 200.0])
    equity, _ = run_rule(
        frame, rule_buy_and_hold, symbol="X", initial_balance=1_000.0, fee_bps=0.0, slippage_bps=0.0
    )
    assert equity.iloc[0] == pytest.approx(1_000.0)  # la primera vela sigue en efectivo


def test_el_cruce_de_emas_entra_y_sale():
    precios = [100.0] * 60 + list(np.linspace(100.0, 200.0, 80)) + list(np.linspace(200.0, 100.0, 80))
    equity, trades = run_rule(
        rampa(precios), make_ema_rule(5, 20), symbol="X", initial_balance=1_000.0,
        fee_bps=0.0, slippage_bps=0.0,
    )
    assert len(trades) >= 1
    assert trades[0].pnl > 0  # captura el tramo alcista


def test_rsi_macd_no_entra_en_un_mercado_plano():
    _, trades = run_rule(
        rampa([100.0] * 200), make_rsi_macd_rule(), symbol="X", initial_balance=1_000.0,
        fee_bps=0.0, slippage_bps=0.0,
    )
    assert trades == []


def test_las_comisiones_reducen_el_resultado_de_la_referencia():
    frame = rampa([100.0, 110.0, 120.0])
    con, _ = run_rule(frame, rule_buy_and_hold, symbol="X", initial_balance=1_000.0, fee_bps=50.0, slippage_bps=20.0)
    sin, _ = run_rule(frame, rule_buy_and_hold, symbol="X", initial_balance=1_000.0, fee_bps=0.0, slippage_bps=0.0)
    assert con.iloc[-1] < sin.iloc[-1]


def test_una_regla_que_nunca_entra_conserva_el_capital():
    equity, trades = run_rule(
        rampa([100.0] * 50), lambda f: pd.Series(False, index=f.index),
        symbol="X", initial_balance=1_000.0,
    )
    assert equity.unique().tolist() == [1_000.0] and trades == []


# ------------------------------------------------------------- cartera

def test_la_cartera_reparte_el_capital_a_partes_iguales(frames):
    resultado = run_baseline(frames, rule_buy_and_hold, name="BH", initial_balance=10_000.0, fee_bps=0.0, slippage_bps=0.0)
    assert resultado.equity_curve.iloc[0] == pytest.approx(10_000.0, rel=1e-6)


def test_las_tres_referencias_se_calculan(frames):
    referencias = standard_baselines(frames, initial_balance=10_000.0)
    assert len(referencias) == 3
    assert all(not r.equity_curve.empty for r in referencias)


def test_comprar_y_mantener_opera_menos_que_rsi_macd(frames):
    referencias = {r.name: r for r in standard_baselines(frames, initial_balance=10_000.0)}
    assert referencias["Comprar y mantener"].trade_count < referencias["RSI>50 + MACD>0"].trade_count


def test_una_cartera_sin_datos_devuelve_metricas_neutras():
    resultado = run_baseline({}, rule_buy_and_hold, name="vacia", initial_balance=1_000.0)
    assert resultado.equity_curve.empty and resultado.metrics.trades == 0


# ------------------------------------------------------------ informes

def test_las_operaciones_se_agrupan_por_mes():
    trades = [make_trade(pnl=10.0, closed_at="2024-01-05"), make_trade(pnl=-5.0, closed_at="2024-01-20"),
              make_trade(pnl=7.0, closed_at="2024-03-02")]
    mensual = trades_per_month(trades)
    assert mensual.set_index("mes").loc["2024-01", "operaciones"] == 2
    assert "2024-02" not in mensual["mes"].tolist()


def test_el_resumen_mensual_cuenta_los_meses_sin_operar():
    trades = [make_trade(pnl=1.0, closed_at="2024-01-05"), make_trade(pnl=1.0, closed_at="2024-04-05")]
    resumen = monthly_activity_summary(trades)
    assert resumen["meses"] == 4 and resumen["meses_sin_operar"] == 2


def test_el_resumen_mensual_sin_operaciones_no_falla():
    assert monthly_activity_summary([])["meses"] == 0


def test_el_corte_por_semestre_reparte_las_operaciones():
    indice = pd.date_range("2024-01-01", periods=365, freq="1D", tz="UTC")
    equity = pd.Series(np.linspace(10_000.0, 12_000.0, 365), index=indice)
    trades = [make_trade(pnl=100.0, closed_at="2024-02-15"), make_trade(pnl=-50.0, closed_at="2024-09-15")]
    semestres = split_by_period(equity, trades, freq="semester")
    assert semestres["periodo"].tolist() == ["2024-H1", "2024-H2"]
    assert semestres["operaciones"].tolist() == [1, 1]


def test_los_semestres_son_naturales_y_comparables_entre_anos():
    """Enero-junio y julio-diciembre, no ventanas desde la primera vela."""
    indice = pd.date_range("2023-04-01", periods=500, freq="1D", tz="UTC")
    equity = pd.Series(np.linspace(1.0, 2.0, 500), index=indice)
    assert split_by_period(equity, [], freq="semester")["periodo"].tolist() == [
        "2023-H1", "2023-H2", "2024-H1", "2024-H2",
    ]


def test_un_periodo_no_soportado_es_un_error():
    from bot.backtesting.reporting import period_label

    with pytest.raises(ValueError, match="no soportado"):
        period_label(pd.Timestamp("2024-01-01"), "decada")


def test_el_corte_por_periodo_de_una_curva_vacia_no_falla():
    assert split_by_period(pd.Series(dtype="float64"), []).empty


def test_la_tabla_comparativa_ordena_las_columnas():
    from bot.backtesting.metrics import Metrics

    tabla = comparison_table({"BOT": (Metrics(total_return_pct=10.0), 5), "BH": (Metrics(total_return_pct=4.0), 1)})
    assert tabla["estrategia"].tolist() == ["BOT", "BH"]
    assert tabla.loc[0, "retorno_%"] == 10.0


def test_las_curvas_se_alinean_en_un_frame():
    a = pd.Series([1.0, 2.0], index=pd.date_range("2024-01-01", periods=2, tz="UTC"))
    b = pd.Series([3.0], index=pd.date_range("2024-01-02", periods=1, tz="UTC"))
    frame = equity_to_frame({"a": a, "b": b})
    assert list(frame.columns) == ["a", "b"] and len(frame) == 2
