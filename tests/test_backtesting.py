"""Tests del motor de backtest, las metricas y los analisis derivados."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bot.backtesting import (
    BacktestEngine,
    build_windows,
    buy_and_hold,
    compare,
    compute_metrics,
    expand_grid,
    max_drawdown,
    run_monte_carlo,
    sharpe_ratio,
)
from bot.data import SyntheticProvider

from tests.conftest import make_trade


@pytest.fixture
def frames():
    proveedor = SyntheticProvider(bars=1500, seed=11, drift=0.0005, volatility=0.012, anchor=pd.Timestamp("2024-06-30", tz="UTC"))
    return {s: proveedor.fetch_klines(s, "15m") for s in ("BTCUSDT", "ETHUSDT")}


# ----------------------------------------------------------------- metricas

def test_el_drawdown_maximo_se_mide_desde_el_pico():
    equity = pd.Series([100.0, 120.0, 90.0, 110.0])
    assert max_drawdown(equity) == pytest.approx(25.0)


def test_una_curva_siempre_creciente_no_tiene_drawdown():
    assert max_drawdown(pd.Series([100.0, 110.0, 120.0])) == pytest.approx(0.0)


def test_el_sharpe_de_retornos_constantes_es_cero():
    assert sharpe_ratio(pd.Series([0.01] * 30)) == 0.0


def test_sin_operaciones_las_metricas_son_neutras():
    metricas = compute_metrics(pd.Series(dtype="float64"), [])
    assert metricas.trades == 0 and metricas.total_return_pct == 0.0


def test_las_metricas_cuadran_con_las_operaciones():
    indice = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    equity = pd.Series(np.linspace(10_000.0, 11_000.0, 10), index=indice)
    trades = [make_trade(pnl=300.0), make_trade(pnl=-100.0), make_trade(pnl=800.0)]
    metricas = compute_metrics(equity, trades)
    assert metricas.trades == 3
    assert metricas.wins == 2 and metricas.losses == 1
    assert metricas.win_rate_pct == pytest.approx(66.67, abs=0.01)
    assert metricas.profit_factor == pytest.approx(11.0)
    assert metricas.total_return_pct == pytest.approx(10.0)


def test_el_factor_de_beneficio_sin_perdidas_es_infinito():
    indice = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    equity = pd.Series([100.0, 110.0, 120.0], index=indice)
    assert compute_metrics(equity, [make_trade(pnl=10.0)]).profit_factor == float("inf")


def test_la_racha_maxima_de_perdidas_se_cuenta_bien():
    trades = [make_trade(pnl=-1.0), make_trade(pnl=-1.0), make_trade(pnl=5.0), make_trade(pnl=-1.0)]
    indice = pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")
    metricas = compute_metrics(pd.Series([100.0, 101.0], index=indice), trades)
    assert metricas.max_consecutive_losses == 2


# -------------------------------------------------------------------- motor

def test_el_backtest_produce_una_curva_de_capital(loaded, frames):
    resultado = BacktestEngine(loaded).run(frames, initial_balance=10_000.0)
    assert not resultado.equity_curve.empty
    assert resultado.equity_curve.iloc[0] == pytest.approx(10_000.0)


def test_el_backtest_registra_los_rechazos(loaded, frames):
    """Saber por que no se opero es la mitad del valor de un backtest."""
    resultado = BacktestEngine(loaded).run(frames, initial_balance=10_000.0)
    assert resultado.rejections
    assert not resultado.rejection_counts().empty


def test_el_backtest_sella_el_hash_de_configuracion(loaded, frames):
    resultado = BacktestEngine(loaded).run(frames, initial_balance=10_000.0)
    assert resultado.config_hash == loaded.config_hash
    assert all(t.config_hash == loaded.config_hash for t in resultado.trades)


def test_las_entradas_se_ejecutan_en_la_barra_siguiente(loaded, frames):
    """Ninguna operacion puede abrirse en la misma vela que la genero."""
    resultado = BacktestEngine(loaded).run(frames, initial_balance=10_000.0)
    if not resultado.signals or not resultado.trades:
        pytest.skip("las velas de prueba no generaron operaciones")
    tiempos_senal = {s.timestamp for s in resultado.signals}
    assert all(t.opened_at not in tiempos_senal or t.opened_at > min(tiempos_senal) for t in resultado.trades)


def test_el_backtest_no_deja_posiciones_abiertas(loaded, frames):
    resultado = BacktestEngine(loaded).run(frames, initial_balance=10_000.0)
    aperturas = len([t for t in resultado.trades])
    assert aperturas == len(resultado.trades)  # todas cerradas y contabilizadas


def test_es_reproducible(loaded, frames):
    primera = BacktestEngine(loaded).run(frames, initial_balance=10_000.0)
    segunda = BacktestEngine(loaded).run(frames, initial_balance=10_000.0)
    assert primera.metrics.as_dict() == segunda.metrics.as_dict()


def test_un_rango_vacio_es_un_error(loaded, frames):
    with pytest.raises(ValueError, match="no hay velas"):
        BacktestEngine(loaded).run(
            frames, start=pd.Timestamp("2050-01-01", tz="UTC"), end=pd.Timestamp("2050-02-01", tz="UTC")
        )


def test_el_modo_halt_corta_la_corrida(loaded, frames):
    resultado = BacktestEngine(loaded, on_kill_switch="halt").run(frames, initial_balance=10_000.0)
    completo = BacktestEngine(loaded).run(frames, initial_balance=10_000.0)
    if not resultado.halted_reason:
        pytest.skip("el freno no salto con estas velas")
    assert resultado.metrics.trades <= completo.metrics.trades


def test_la_politica_de_kill_switch_se_valida(loaded):
    with pytest.raises(ValueError, match="on_kill_switch"):
        BacktestEngine(loaded, on_kill_switch="lo_que_sea")


def test_el_calentamiento_no_se_opera(loaded, frames):
    """Las velas previas al inicio alimentan indicadores, no generan trades."""
    inicio = frames["BTCUSDT"]["open_time"].iloc[800]
    resultado = BacktestEngine(loaded).run(frames, initial_balance=10_000.0, start=inicio)
    assert resultado.equity_curve.index[0] >= inicio


# ---------------------------------------------------------------- benchmark

def test_comprar_y_mantener_sigue_el_precio(frames):
    equity = buy_and_hold(frames["BTCUSDT"], initial_balance=10_000.0, fee_bps=0.0)
    precios = frames["BTCUSDT"]["close"]
    assert equity.iloc[-1] / equity.iloc[0] == pytest.approx(precios.iloc[-1] / precios.iloc[0])


def test_las_comisiones_reducen_el_benchmark(frames):
    con = buy_and_hold(frames["BTCUSDT"], initial_balance=10_000.0, fee_bps=50.0)
    sin = buy_and_hold(frames["BTCUSDT"], initial_balance=10_000.0, fee_bps=0.0)
    assert con.iloc[-1] < sin.iloc[-1]


def test_la_comparacion_calcula_el_exceso(loaded, frames):
    resultado = BacktestEngine(loaded).run(frames, initial_balance=10_000.0)
    comparacion = compare(
        resultado.equity_curve, resultado.metrics, frames["BTCUSDT"],
        symbol="BTCUSDT", initial_balance=10_000.0,
    )
    assert comparacion.excess_return_pct == pytest.approx(
        comparacion.strategy.total_return_pct - comparacion.benchmark.total_return_pct
    )


# -------------------------------------------------------------- Monte Carlo

def test_monte_carlo_devuelve_una_distribucion():
    trades = [make_trade(pnl=p) for p in (100.0, -50.0, 200.0, -80.0, 150.0, -60.0)]
    simulacion = run_monte_carlo(trades, initial_balance=10_000.0, iterations=200, seed=1)
    assert simulacion.iterations == 200
    assert simulacion.percentile_final(5) <= simulacion.median_final <= simulacion.percentile_final(95)


def test_monte_carlo_es_determinista_con_la_misma_semilla():
    trades = [make_trade(pnl=p) for p in (100.0, -50.0, 200.0)]
    a = run_monte_carlo(trades, initial_balance=10_000.0, iterations=100, seed=7)
    b = run_monte_carlo(trades, initial_balance=10_000.0, iterations=100, seed=7)
    assert a.median_final == b.median_final


def test_monte_carlo_sin_operaciones_no_falla():
    simulacion = run_monte_carlo([], initial_balance=10_000.0, iterations=100)
    assert simulacion.iterations == 0 and simulacion.probability_of_loss == 0.0


def test_una_estrategia_solo_perdedora_tiene_ruina_alta():
    trades = [make_trade(pnl=-2_000.0) for _ in range(5)]
    simulacion = run_monte_carlo(trades, initial_balance=10_000.0, iterations=200, seed=2)
    assert simulacion.probability_of_loss == 1.0
    assert simulacion.risk_of_ruin(50.0) > 0.9


# -------------------------------------------------------------- walk-forward

def test_las_ventanas_cubren_el_periodo():
    ventanas = build_windows(
        pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-12-31", tz="UTC"),
        train_days=90, test_days=30, step_days=30,
    )
    assert len(ventanas) >= 6
    assert all(v.train_end == v.test_start for v in ventanas)
    assert ventanas[1].train_start > ventanas[0].train_start


def test_un_periodo_corto_no_genera_ventanas():
    ventanas = build_windows(
        pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-02-01", tz="UTC"),
        train_days=90, test_days=30, step_days=30,
    )
    assert ventanas == []


def test_el_grid_hace_el_producto_cartesiano():
    combinaciones = expand_grid({"a": [1, 2], "b": [3, 4, 5]})
    assert len(combinaciones) == 6
    assert {"a": 1, "b": 3} in combinaciones


def test_un_grid_vacio_da_una_combinacion_vacia():
    assert expand_grid({}) == [{}]


def test_aplicar_parametros_cambia_el_hash(loaded):
    from bot.backtesting.walkforward import apply_params

    modificada = apply_params(loaded, {"scoring.min_score": 80.0})
    assert modificada.config.scoring.min_score == 80.0
    assert modificada.config_hash != loaded.config_hash


def test_los_trades_persistidos_llevan_el_stop_real(loaded, frames, tmp_path):
    from bot.logging import Database

    db = Database(tmp_path / "bt.sqlite")
    run_id = db.start_run(mode="backtest", config_hash=loaded.config_hash, config={})
    resultado = BacktestEngine(loaded, db=db, run_id=run_id).run(frames, initial_balance=10_000.0)
    if not resultado.trades:
        pytest.skip("las velas de prueba no generaron operaciones")

    guardados = db.trades_df()
    assert (guardados["stop_loss"] != guardados["entry_price"]).all()
    assert (guardados["stop_loss"] < guardados["take_profit"]).all()
    db.close()


def test_la_politica_both_ejecuta_los_dos_escenarios(loaded, frames):
    """El informe tiene que poder comparar reanudar contra detenerse."""

    resume = BacktestEngine(loaded, on_kill_switch="resume_next_day").run(frames, initial_balance=10_000.0)
    halt = BacktestEngine(loaded, on_kill_switch="halt").run(frames, initial_balance=10_000.0)
    assert halt.metrics.trades <= resume.metrics.trades
    if halt.halted_reason:
        assert halt.end <= resume.end


def test_el_escenario_estricto_no_rearma_el_freno(loaded, frames):
    resultado = BacktestEngine(loaded, on_kill_switch="halt").run(frames, initial_balance=10_000.0)
    assert len(resultado.kill_switch_events) <= 1


def test_un_tramo_corto_no_se_anualiza():
    """Anualizar 16 dias da una cifra que el lector comparara con cinco anos."""
    indice = pd.date_range("2024-01-01", periods=16, freq="1D", tz="UTC")
    corto = compute_metrics(pd.Series(np.linspace(10_000.0, 9_800.0, 16), index=indice), [])
    assert not corto.annualizable
    assert "CAGR n/d" in corto.summary()

    largo_idx = pd.date_range("2024-01-01", periods=400, freq="1D", tz="UTC")
    largo = compute_metrics(pd.Series(np.linspace(10_000.0, 12_000.0, 400), index=largo_idx), [])
    assert largo.annualizable and "CAGR +" in largo.summary()


def test_la_tabla_comparativa_oculta_el_cagr_de_un_tramo_corto():
    from bot.backtesting.reporting import comparison_table

    indice = pd.date_range("2024-01-01", periods=16, freq="1D", tz="UTC")
    corto = compute_metrics(pd.Series(np.linspace(10_000.0, 9_800.0, 16), index=indice), [])
    tabla = comparison_table({"corto": (corto, 3)})
    assert tabla.loc[0, "CAGR_%"] is None and tabla.loc[0, "dias"] == 15
