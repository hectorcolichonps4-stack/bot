"""Tests de la estrategia: gates, puntuacion y motor."""

from __future__ import annotations

import pytest

from bot.config.schema import GatesConfig, ScoringConfig
from bot.execution.base import Side
from bot.strategy import StrategyEngine, first_failure, run_gates
from bot.strategy.gates import (
    gate_room_to_target,
    gate_rsi,
    gate_trend,
    gate_volatility,
    gate_volume,
)
from bot.strategy.scoring import Scorer, score_rsi, score_volume

from tests.conftest import make_snapshot

GATES = GatesConfig()


# -------------------------------------------------------------------- gates

def test_una_senal_limpia_pasa_todos_los_gates():
    resultados = run_gates(make_snapshot(), Side.LONG, GATES)
    assert first_failure(resultados) is None


def test_sin_alineacion_de_medias_no_hay_senal():
    snap = make_snapshot(ema_fast=95.0, ema_slow=97.0, ema_trend=99.0)
    assert not gate_trend(snap, Side.LONG, GATES).passed


def test_el_volumen_bajo_bloquea():
    assert not gate_volume(make_snapshot(relative_volume=0.5), Side.LONG, GATES).passed


def test_la_liquidez_insuficiente_bloquea():
    resultado = gate_volume(make_snapshot(quote_volume=1_000.0), Side.LONG, GATES)
    assert not resultado.passed and resultado.name == "liquidity"


def test_el_rsi_sobrecomprado_bloquea_los_largos():
    assert not gate_rsi(make_snapshot(rsi=80.0), Side.LONG, GATES).passed


def test_el_rsi_sobrecomprado_no_bloquea_los_cortos():
    assert gate_rsi(make_snapshot(rsi=80.0), Side.SHORT, GATES).passed


def test_la_volatilidad_fuera_de_rango_bloquea():
    assert not gate_volatility(make_snapshot(atr_pct=0.1), Side.LONG, GATES).passed
    assert not gate_volatility(make_snapshot(atr_pct=15.0), Side.LONG, GATES).passed


def test_sin_recorrido_hasta_la_resistencia_no_se_entra():
    from bot.indicators import Levels

    snap = make_snapshot(close=100.0, atr=5.0, levels=Levels(supports=(90.0,), resistances=(101.0,)))
    assert not gate_room_to_target(snap, Side.LONG, GATES).passed


def test_los_cortos_estan_desactivados_por_defecto():
    resultados = run_gates(make_snapshot(ema_fast=95.0, ema_slow=97.0, ema_trend=99.0), Side.SHORT, GATES)
    assert any(r.name == "side_allowed" and not r.passed for r in resultados)


def test_la_lista_negra_bloquea_el_simbolo():
    gates = GatesConfig(blacklist=["BTCUSDT"])
    assert not first_failure(run_gates(make_snapshot(), Side.LONG, gates)) is None


def test_los_gates_reportan_todos_los_fallos_a_la_vez():
    """Ver todos los motivos de golpe ahorra iteraciones al afinar."""
    snap = make_snapshot(relative_volume=0.2, rsi=90.0, atr_pct=0.01)
    fallos = [r.name for r in run_gates(snap, Side.LONG, GATES) if not r.passed]
    assert {"volume", "rsi", "volatility"} <= set(fallos)


def test_los_gates_desactivados_dejan_pasar_todo():
    gates = GatesConfig(enabled=False)
    snap = make_snapshot(relative_volume=0.0, rsi=99.0)
    assert first_failure(run_gates(snap, Side.LONG, gates)) is None


# --------------------------------------------------------------- puntuacion

def test_la_puntuacion_esta_entre_0_y_100():
    breakdown = Scorer(ScoringConfig()).score(make_snapshot(), Side.LONG)
    assert 0.0 <= breakdown.total <= 100.0


def test_los_pesos_se_normalizan_a_100():
    """Duplicar todos los pesos no debe cambiar la escala del umbral."""
    base = ScoringConfig()
    doble = ScoringConfig(weights={k: v * 2 for k, v in base.weights.items()})
    snap = make_snapshot()
    assert Scorer(base).score(snap, Side.LONG).total == pytest.approx(
        Scorer(doble).score(snap, Side.LONG).total
    )


def test_una_configuracion_perfecta_llega_a_100():
    from bot.indicators import Levels

    snap = make_snapshot(
        ema_fast=110.0, ema_slow=100.0, atr=1.0, macd_hist=1.0, macd_hist_prev=0.0,
        rsi=55.0, relative_volume=3.0, atr_pct=1.5,
        levels=Levels(supports=(80.0,), resistances=(200.0,)),
    )
    assert Scorer(ScoringConfig()).score(snap, Side.LONG).total == pytest.approx(100.0)


def test_un_peso_sin_componente_es_un_error():
    with pytest.raises(ValueError, match="pesos sin componente"):
        Scorer(ScoringConfig(weights={"inexistente": 1.0}))


def test_el_componente_de_rsi_premia_la_zona_media():
    assert score_rsi(make_snapshot(rsi=55.0), Side.LONG) == 1.0
    assert score_rsi(make_snapshot(rsi=85.0), Side.LONG) == 0.0


def test_el_componente_de_volumen_se_satura():
    assert score_volume(make_snapshot(relative_volume=3.0), Side.LONG) == 1.0
    assert score_volume(make_snapshot(relative_volume=10.0), Side.LONG) == 1.0


def test_el_desglose_suma_el_total():
    breakdown = Scorer(ScoringConfig()).score(make_snapshot(), Side.LONG)
    assert sum(breakdown.weighted.values()) == pytest.approx(breakdown.total)


# -------------------------------------------------------------------- motor

def test_el_motor_devuelve_rechazo_sin_calentamiento(loaded, candles):
    motor = StrategyEngine(loaded)
    decision = motor.evaluate("BTCUSDT", motor.prepare(candles), index=10)
    assert not decision.accepted and decision.rejection.stage == "data"


def test_el_motor_siempre_da_un_motivo(loaded, candles):
    motor = StrategyEngine(loaded)
    decision = motor.evaluate("BTCUSDT", motor.prepare(candles))
    assert decision.accepted or decision.rejection.reason


def test_la_senal_lleva_el_hash_de_la_configuracion(loaded, candles):
    """Sin el hash, un resultado historico no se puede reatribuir."""
    motor = StrategyEngine(loaded)
    features = motor.prepare(candles)
    for index in range(300, len(features)):
        decision = motor.evaluate("BTCUSDT", features, index=index)
        if decision.accepted:
            assert decision.signal.config_hash == loaded.config_hash
            return
    pytest.skip("las velas de prueba no generaron ninguna senal")


def test_los_niveles_de_salida_respetan_el_lado(loaded, candles):
    motor = StrategyEngine(loaded)
    features = motor.prepare(candles)
    for index in range(300, len(features)):
        decision = motor.evaluate("BTCUSDT", features, index=index)
        if decision.accepted:
            sig = decision.signal
            assert sig.stop_loss < sig.entry_price < sig.take_profit
            assert sig.reward_risk == pytest.approx(loaded.config.risk.take_profit_r_multiple)
            return
    pytest.skip("las velas de prueba no generaron ninguna senal")


def test_evaluate_y_evaluate_at_coinciden(loaded, candles):
    motor = StrategyEngine(loaded)
    features = motor.prepare(candles)
    builder = motor.builder("BTCUSDT", features)
    directa = motor.evaluate("BTCUSDT", features, index=400)
    reutilizada = motor.evaluate_at(builder, 400)
    assert directa.accepted == reutilizada.accepted


def test_el_cuadro_de_indicadores_no_muta_las_velas(loaded, candles):
    columnas = list(candles.columns)
    StrategyEngine(loaded).prepare(candles)
    assert list(candles.columns) == columnas
