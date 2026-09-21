"""Tests de gestion de riesgo.

Es el modulo que decide cuanto se arriesga y cuando hay que parar, asi que se
prueba a conciencia: tamano de posicion, limites diarios y semanales,
cooldowns, correlacion, kill switch y el orden en que el gestor los aplica.
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from bot.data.filters import SymbolFilters
from bot.execution.base import AccountState, Position, Side
from bot.risk import (
    KillSwitch,
    Ledger,
    RiskManager,
    check_concurrent_positions,
    check_correlation,
    check_daily_loss,
    check_daily_trades,
    check_exposure,
    check_global_cooldown,
    check_symbol_cooldown,
    check_weekly_loss,
    correlation_matrix,
    day_start,
    pairwise_correlation,
    position_size,
    week_start,
)
from bot.strategy.signals import Signal

from tests.conftest import make_trade

PERMISSIVE = SymbolFilters.permissive("BTCUSDT")
NOW = pd.Timestamp("2024-06-05 12:00", tz="UTC")  # miercoles


def account(equity: float = 10_000.0, cash: float | None = None, positions=()) -> AccountState:
    return AccountState(cash=equity if cash is None else cash, equity=equity, positions=tuple(positions))


def position(symbol: str = "ETHUSDT", notional: float = 1_000.0) -> Position:
    return Position(
        symbol=symbol, side=Side.LONG, quantity=notional / 100.0, entry_price=100.0,
        stop_loss=95.0, take_profit=110.0, opened_at=NOW,
    )


def signal(symbol: str = "BTCUSDT", entry: float = 100.0, stop: float = 95.0) -> Signal:
    return Signal(
        symbol=symbol, side=Side.LONG, timestamp=NOW, entry_price=entry,
        stop_loss=stop, take_profit=entry + (entry - stop) * 2,
        score=75.0, components={}, snapshot={}, config_hash="test",
    )


# ------------------------------------------------------ tamano de posicion

def test_el_tamano_arriesga_exactamente_el_porcentaje_configurado():
    """1% de 10.000 con un stop a 5 unidades son 20 unidades."""
    resultado = position_size(
        equity=10_000.0, entry_price=100.0, stop_loss=95.0, risk_per_trade_pct=1.0,
        max_position_pct=100.0, available_cash=10_000.0, filters=PERMISSIVE,
    )
    assert resultado.quantity == pytest.approx(20.0)
    assert resultado.risk_amount == pytest.approx(100.0)


def test_un_stop_mas_ajustado_permite_mas_cantidad_con_el_mismo_riesgo():
    ancho = position_size(
        equity=10_000.0, entry_price=100.0, stop_loss=90.0, risk_per_trade_pct=1.0,
        max_position_pct=100.0, available_cash=10_000.0, filters=PERMISSIVE,
    )
    estrecho = position_size(
        equity=10_000.0, entry_price=100.0, stop_loss=98.0, risk_per_trade_pct=1.0,
        max_position_pct=100.0, available_cash=10_000.0, filters=PERMISSIVE,
    )
    assert estrecho.quantity > ancho.quantity
    assert estrecho.risk_amount == pytest.approx(ancho.risk_amount)


def test_el_tope_de_posicion_recorta_el_tamano():
    resultado = position_size(
        equity=10_000.0, entry_price=100.0, stop_loss=99.0, risk_per_trade_pct=1.0,
        max_position_pct=20.0, available_cash=10_000.0, filters=PERMISSIVE,
    )
    assert resultado.notional == pytest.approx(2_000.0)
    assert resultado.capped_by == "max_position_pct"


def test_el_efectivo_disponible_recorta_el_tamano():
    resultado = position_size(
        equity=10_000.0, entry_price=100.0, stop_loss=99.0, risk_per_trade_pct=1.0,
        max_position_pct=100.0, available_cash=500.0, filters=PERMISSIVE,
    )
    assert resultado.notional == pytest.approx(500.0)
    assert resultado.capped_by == "available_cash"


def test_un_stop_igual_a_la_entrada_no_es_operable():
    resultado = position_size(
        equity=10_000.0, entry_price=100.0, stop_loss=100.0, risk_per_trade_pct=1.0,
        max_position_pct=100.0, available_cash=10_000.0, filters=PERMISSIVE,
    )
    assert not resultado.viable and resultado.capped_by == "stop"


def test_el_redondeo_al_step_nunca_aumenta_el_riesgo():
    """Truncar hacia abajo: redondear hacia arriba excederia lo autorizado."""
    filtros = SymbolFilters("BTCUSDT", step_size=Decimal("0.1"), min_notional=Decimal("0"))
    resultado = position_size(
        equity=10_000.0, entry_price=100.0, stop_loss=95.0, risk_per_trade_pct=1.0,
        max_position_pct=100.0, available_cash=10_000.0, filters=filtros,
    )
    assert resultado.quantity == pytest.approx(20.0)
    assert resultado.risk_amount <= 100.0 + 1e-9


def test_si_el_step_anula_la_cantidad_no_hay_operacion():
    filtros = SymbolFilters("BTCUSDT", step_size=Decimal("1000"), min_notional=Decimal("0"))
    resultado = position_size(
        equity=100.0, entry_price=100.0, stop_loss=95.0, risk_per_trade_pct=1.0,
        max_position_pct=100.0, available_cash=100.0, filters=filtros,
    )
    assert not resultado.viable and resultado.capped_by == "step_size"


def test_un_notional_por_debajo_del_minimo_se_rechaza():
    filtros = SymbolFilters("BTCUSDT", step_size=Decimal("0.00001"), min_notional=Decimal("50"))
    resultado = position_size(
        equity=100.0, entry_price=100.0, stop_loss=95.0, risk_per_trade_pct=1.0,
        max_position_pct=100.0, available_cash=100.0, filters=filtros,
    )
    assert not resultado.viable and resultado.capped_by == "exchange_filters"


def test_sin_equity_no_se_opera():
    resultado = position_size(
        equity=0.0, entry_price=100.0, stop_loss=95.0, risk_per_trade_pct=1.0,
        max_position_pct=100.0, available_cash=0.0, filters=PERMISSIVE,
    )
    assert not resultado.viable


# ------------------------------------------------------------------ limites

def test_la_perdida_diaria_bloquea_al_alcanzar_el_limite(config):
    ledger = Ledger(trades=[make_trade(pnl=-310.0, closed_at=NOW - pd.Timedelta(hours=2))])
    check = check_daily_loss(ledger, 10_000.0, NOW, config.risk)  # limite 3% = 300
    assert not check.passed


def test_la_perdida_diaria_no_bloquea_por_debajo_del_limite(config):
    ledger = Ledger(trades=[make_trade(pnl=-100.0, closed_at=NOW - pd.Timedelta(hours=2))])
    assert check_daily_loss(ledger, 10_000.0, NOW, config.risk).passed


def test_las_perdidas_de_ayer_no_cuentan_para_hoy(config):
    ledger = Ledger(trades=[make_trade(pnl=-900.0, closed_at=NOW - pd.Timedelta(days=1))])
    assert check_daily_loss(ledger, 10_000.0, NOW, config.risk).passed


def test_la_perdida_semanal_acumula_varios_dias(config):
    ledger = Ledger(trades=[
        make_trade(pnl=-300.0, closed_at=NOW - pd.Timedelta(days=2)),
        make_trade(pnl=-250.0, closed_at=NOW - pd.Timedelta(days=1)),
        make_trade(pnl=-200.0, closed_at=NOW - pd.Timedelta(hours=1)),
    ])
    assert not check_weekly_loss(ledger, 10_000.0, NOW, config.risk).passed  # limite 7% = 700


def test_las_ganancias_compensan_dentro_del_periodo(config):
    ledger = Ledger(trades=[
        make_trade(pnl=-400.0, closed_at=NOW - pd.Timedelta(hours=3)),
        make_trade(pnl=300.0, closed_at=NOW - pd.Timedelta(hours=1)),
    ])
    assert check_daily_loss(ledger, 10_000.0, NOW, config.risk).passed


def test_el_limite_de_operaciones_diarias_corta(config):
    ledger = Ledger(trades=[make_trade(pnl=1.0, closed_at=NOW) for _ in range(8)])
    assert not check_daily_trades(ledger, NOW, config.risk).passed


def test_el_limite_de_posiciones_concurrentes_corta(config):
    assert check_concurrent_positions(3, config.risk).passed is False
    assert check_concurrent_positions(2, config.risk).passed is True


def test_la_exposicion_agregada_se_controla(config):
    assert check_exposure(50.0, 20.0, config.risk).passed is False  # maximo 60%
    assert check_exposure(30.0, 20.0, config.risk).passed is True


def test_inicio_de_dia_y_de_semana():
    assert day_start(NOW) == pd.Timestamp("2024-06-05 00:00", tz="UTC")
    assert week_start(NOW) == pd.Timestamp("2024-06-03 00:00", tz="UTC")  # lunes


# --------------------------------------------------------------- cooldowns

def test_tras_una_perdida_se_pausa_toda_la_operativa(config):
    ledger = Ledger(trades=[make_trade(pnl=-50.0, closed_at=NOW - pd.Timedelta(minutes=10))])
    check = check_global_cooldown(ledger, NOW, config.risk)  # 60 minutos
    assert not check.passed and check.remaining_minutes == pytest.approx(50.0)


def test_el_enfriamiento_global_expira(config):
    ledger = Ledger(trades=[make_trade(pnl=-50.0, closed_at=NOW - pd.Timedelta(minutes=61))])
    assert check_global_cooldown(ledger, NOW, config.risk).passed


def test_una_ganancia_no_activa_el_enfriamiento(config):
    ledger = Ledger(trades=[make_trade(pnl=50.0, closed_at=NOW - pd.Timedelta(minutes=1))])
    assert check_global_cooldown(ledger, NOW, config.risk).passed


def test_el_mismo_simbolo_tiene_su_propio_enfriamiento(config):
    ledger = Ledger(trades=[make_trade(pnl=50.0, symbol="BTCUSDT", closed_at=NOW - pd.Timedelta(minutes=30))])
    assert not check_symbol_cooldown(ledger, "BTCUSDT", NOW, config.risk).passed  # 120 minutos
    assert check_symbol_cooldown(ledger, "ETHUSDT", NOW, config.risk).passed


# -------------------------------------------------------------- correlacion

def test_dos_series_identicas_correlacionan_uno():
    serie = pd.Series(np.linspace(1.0, 2.0, 50) + np.sin(np.arange(50)))
    assert pairwise_correlation(serie, serie) == pytest.approx(1.0)


def test_una_serie_plana_no_correlaciona():
    plana = pd.Series([1.0] * 50)
    variable = pd.Series(np.arange(50, dtype="float64"))
    assert pairwise_correlation(plana, variable) == 0.0


def test_se_rechaza_una_senal_muy_correlacionada_con_lo_abierto():
    rng = np.random.default_rng(3)
    base = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 200))))
    closes = {"BTCUSDT": base, "ETHUSDT": base * 1.5}  # misma forma, distinta escala
    check = check_correlation("BTCUSDT", ["ETHUSDT"], closes, lookback=120, max_correlation=0.85)
    assert not check.passed and check.worst_symbol == "ETHUSDT"


def test_se_acepta_una_senal_poco_correlacionada():
    rng = np.random.default_rng(4)
    closes = {
        "BTCUSDT": pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))),
        "ETHUSDT": pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))),
    }
    assert check_correlation("BTCUSDT", ["ETHUSDT"], closes, lookback=120, max_correlation=0.85).passed


def test_la_correlacion_negativa_tambien_concentra_riesgo():
    """-0.95 concentra tanto como +0.95: se compara en valor absoluto."""
    rng = np.random.default_rng(5)
    retornos = rng.normal(0, 0.01, 200)
    closes = {
        "BTCUSDT": pd.Series(100.0 * np.exp(np.cumsum(retornos))),
        "ETHUSDT": pd.Series(100.0 * np.exp(np.cumsum(-retornos))),
    }
    assert not check_correlation("BTCUSDT", ["ETHUSDT"], closes, lookback=120, max_correlation=0.85).passed


def test_sin_posiciones_abiertas_no_hay_nada_que_correlacionar():
    assert check_correlation("BTCUSDT", [], {}, lookback=120, max_correlation=0.85).passed


def test_la_matriz_de_correlacion_tiene_diagonal_uno():
    rng = np.random.default_rng(6)
    closes = {s: pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 100)))) for s in ("A", "B")}
    matriz = correlation_matrix(closes, 50)
    assert matriz.loc["A", "A"] == pytest.approx(1.0)
    assert matriz.loc["A", "B"] == pytest.approx(matriz.loc["B", "A"])


# -------------------------------------------------------------- kill switch

def test_la_racha_de_perdidas_dispara_el_freno(config):
    switch = KillSwitch(config.risk.kill_switch)
    ledger = Ledger(trades=[make_trade(pnl=-10.0) for _ in range(4)])
    assert switch.evaluate(ledger, 10_000.0, NOW) is not None
    assert switch.active


def test_una_ganancia_rompe_la_racha(config):
    switch = KillSwitch(config.risk.kill_switch)
    ledger = Ledger(trades=[make_trade(pnl=-10.0), make_trade(pnl=-10.0), make_trade(pnl=5.0), make_trade(pnl=-10.0)])
    assert switch.evaluate(ledger, 10_000.0, NOW) is None


def test_el_drawdown_dispara_el_freno(config):
    switch = KillSwitch(config.risk.kill_switch)
    ledger = Ledger(equity_high_water=10_000.0)
    assert switch.evaluate(ledger, 8_400.0, NOW) is not None  # -16%, limite 15%


def test_la_tasa_de_errores_dispara_el_freno(config):
    switch = KillSwitch(config.risk.kill_switch)
    for _ in range(10):
        switch.record_error(NOW)
    assert switch.evaluate(Ledger(), 10_000.0, NOW) is not None


def test_los_errores_antiguos_no_cuentan(config):
    switch = KillSwitch(config.risk.kill_switch)
    for _ in range(10):
        switch.record_error(NOW - pd.Timedelta(hours=2))
    assert switch.evaluate(Ledger(), 10_000.0, NOW) is None


def test_el_freno_no_se_desactiva_solo(config):
    """Rearmarlo es una decision humana, no algo que el bot haga por su cuenta."""
    switch = KillSwitch(config.risk.kill_switch)
    switch.force("prueba", NOW)
    ledger = Ledger(trades=[make_trade(pnl=100.0)])
    assert switch.evaluate(ledger, 20_000.0, NOW) is not None
    switch.reset()
    assert not switch.active


def test_el_freno_desactivado_no_dispara():
    from bot.config.schema import KillSwitchConfig

    switch = KillSwitch(KillSwitchConfig(enabled=False))
    ledger = Ledger(trades=[make_trade(pnl=-10.0) for _ in range(10)])
    assert switch.evaluate(ledger, 1.0, NOW) is None


def test_el_rearme_no_vuelve_a_disparar_en_el_acto(loaded):
    """Sin limpiar la racha, rearmar seria inutil: las perdidas siguen ahi."""
    risk = RiskManager(loaded)
    for _ in range(4):
        risk.ledger.record(make_trade(pnl=-10.0))
    assert risk.kill_switch.evaluate(risk.ledger, 10_000.0, NOW) is not None

    risk.rearm_kill_switch(10_000.0)
    assert risk.kill_switch.evaluate(risk.ledger, 10_000.0, NOW) is None


# ------------------------------------------------------------ RiskManager

def test_el_gestor_aprueba_una_senal_limpia(loaded):
    decision = RiskManager(loaded).evaluate(signal(), account(), filters=PERMISSIVE, now=NOW)
    assert decision.approved and decision.quantity > 0


def test_el_gestor_no_duplica_posicion_en_el_mismo_simbolo(loaded):
    estado = account(positions=[position("BTCUSDT")])
    decision = RiskManager(loaded).evaluate(signal("BTCUSDT"), estado, filters=PERMISSIVE, now=NOW)
    assert decision.rejected and decision.stage == "duplicate"


def test_el_kill_switch_tiene_prioridad_sobre_todo(loaded):
    risk = RiskManager(loaded)
    risk.kill_switch.force("parada manual", NOW)
    decision = risk.evaluate(signal(), account(), filters=PERMISSIVE, now=NOW)
    assert decision.rejected and decision.stage == "kill_switch"


def test_el_gestor_respeta_el_limite_diario(loaded):
    risk = RiskManager(loaded)
    risk.ledger.record(make_trade(pnl=-400.0, closed_at=NOW - pd.Timedelta(hours=2)))
    decision = risk.evaluate(signal(), account(), filters=PERMISSIVE, now=NOW)
    assert decision.rejected and decision.stage == "daily_loss"


def test_el_gestor_respeta_el_enfriamiento(loaded):
    risk = RiskManager(loaded)
    risk.ledger.record(make_trade(pnl=-10.0, symbol="ETHUSDT", closed_at=NOW - pd.Timedelta(minutes=5)))
    decision = risk.evaluate(signal(), account(), filters=PERMISSIVE, now=NOW)
    assert decision.rejected and decision.stage == "cooldown_after_loss"


def test_el_gestor_registra_los_controles_superados(loaded):
    decision = RiskManager(loaded).evaluate(signal(), account(), filters=PERMISSIVE, now=NOW)
    assert "kill_switch" in decision.checks and "correlation" in decision.checks


def test_al_cerrar_un_trade_se_actualiza_el_historial(loaded):
    risk = RiskManager(loaded)
    risk.on_trade_closed(make_trade(pnl=-50.0), 9_950.0, NOW)
    assert len(risk.ledger.trades) == 1
    assert risk.ledger.consecutive_losses() == 1
