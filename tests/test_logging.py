"""Tests de la persistencia en SQLite."""

from __future__ import annotations

import pandas as pd
import pytest

from bot.execution.base import Side
from bot.logging import Database
from bot.strategy.signals import Signal

NOW = pd.Timestamp("2024-06-05 12:00", tz="UTC")


@pytest.fixture
def db(tmp_path) -> Database:
    base = Database(tmp_path / "bot.sqlite")
    yield base
    base.close()


def signal(symbol: str = "BTCUSDT") -> Signal:
    return Signal(
        symbol=symbol, side=Side.LONG, timestamp=NOW, entry_price=100.0, stop_loss=95.0,
        take_profit=110.0, score=72.5, components={"trend": 0.8}, snapshot={"rsi": 55.0},
        config_hash="abc1234567890def",
    )


def test_el_esquema_se_crea_solo(db):
    tablas = db.query("SELECT name FROM sqlite_master WHERE type='table'")["name"].tolist()
    assert {"runs", "signals", "rejections", "trades", "errors", "bot_state", "equity_curve"} <= set(tablas)


def test_una_senal_se_guarda_con_su_hash(db):
    run_id = db.start_run(mode="paper", config_hash="abc1234567890def", config={})
    db.log_signal(signal(), run_id=run_id)
    fila = db.signals_df().iloc[0]
    assert fila["symbol"] == "BTCUSDT" and fila["config_hash"] == "abc1234567890def"
    assert fila["acted_on"] == 0


def test_marcar_una_senal_como_ejecutada(db):
    signal_id = db.log_signal(signal())
    db.mark_signal_acted(signal_id)
    assert db.signals_df().iloc[0]["acted_on"] == 1


def test_un_rechazo_conserva_el_motivo_y_el_detalle(db):
    db.log_rejection(
        symbol="ETHUSDT", bar_time=NOW, stage="gate",
        reason="[volume] volumen relativo 0.40 < 1.2",
        detail={"relative_volume": 0.4}, config_hash="h",
    )
    fila = db.rejections_df().iloc[0]
    assert fila["stage"] == "gate" and "volumen relativo" in fila["reason"]
    assert "relative_volume" in fila["detail"]


def test_el_resumen_de_rechazos_agrupa(db):
    for _ in range(3):
        db.log_rejection(symbol="BTCUSDT", bar_time=NOW, stage="gate", reason="sin tendencia", config_hash="h")
    db.log_rejection(symbol="BTCUSDT", bar_time=NOW, stage="risk", reason="limite diario", config_hash="h")
    resumen = db.rejection_summary()
    assert resumen.iloc[0]["veces"] == 3


def test_el_ciclo_de_un_trade_se_registra_entero(db):
    trade_id = db.open_trade(
        symbol="BTCUSDT", side="long", quantity=0.5, entry_price=100.0, stop_loss=95.0,
        take_profit=110.0, opened_at=NOW, mode="paper", config_hash="h",
    )
    abierto = db.trades_df().iloc[0]
    assert pd.isna(abierto["closed_at"]) and pd.isna(abierto["pnl"])

    db.close_trade(
        trade_id, exit_price=110.0, closed_at=NOW + pd.Timedelta(hours=1),
        pnl=5.0, fees=0.1, r_multiple=2.0, exit_reason="take_profit",
    )
    cerrado = db.trades_df().iloc[0]
    assert cerrado["pnl"] == 5.0 and cerrado["exit_reason"] == "take_profit"


def test_los_trades_se_filtran_por_hash_de_configuracion(db):
    """Es lo que permite comparar resultados entre dos configuraciones."""
    for hash_config in ("aaa", "bbb", "aaa"):
        db.open_trade(
            symbol="BTCUSDT", side="long", quantity=1.0, entry_price=100.0, stop_loss=95.0,
            take_profit=110.0, opened_at=NOW, mode="backtest", config_hash=hash_config,
        )
    assert len(db.trades_df(config_hash="aaa")) == 2


def test_los_errores_guardan_la_traza(db):
    try:
        raise ValueError("fallo de prueba")
    except ValueError as exc:
        db.log_error(source="test", message="algo fallo", exc=exc)
    fila = db.errors_df().iloc[0]
    assert "ValueError" in fila["traceback"] and fila["level"] == "ERROR"


def test_el_estado_va_y_vuelve(db):
    db.set_state("kill_switch", {"active": True, "reason": "racha"})
    assert db.get_state("kill_switch")["active"] is True
    db.set_state("kill_switch", {"active": False})
    assert db.get_state("kill_switch")["active"] is False  # se sobrescribe, no duplica


def test_una_clave_de_estado_inexistente_devuelve_el_defecto(db):
    assert db.get_state("no_existe", "defecto") == "defecto"


def test_la_curva_de_capital_se_acumula(db):
    run_id = db.start_run(mode="backtest", config_hash="h", config={})
    for i in range(5):
        db.record_equity(equity=10_000.0 + i, cash=10_000.0, recorded_at=NOW + pd.Timedelta(hours=i), run_id=run_id)
    assert len(db.equity_df(run_id=run_id)) == 5


def test_el_modo_solo_lectura_impide_escribir(tmp_path):
    ruta = tmp_path / "bot.sqlite"
    Database(ruta).close()
    lectura = Database(ruta, read_only=True)
    with pytest.raises(RuntimeError, match="solo lectura"):
        lectura.set_state("x", 1)
    lectura.close()


def test_el_modo_solo_lectura_exige_que_exista(tmp_path):
    with pytest.raises(FileNotFoundError):
        Database(tmp_path / "no_existe.sqlite", read_only=True)


def test_una_ejecucion_registra_inicio_y_fin(db):
    run_id = db.start_run(mode="live", config_hash="h", config={"a": 1}, notes="prueba")
    db.finish_run(run_id)
    fila = db.query("SELECT * FROM runs WHERE id = ?", (run_id,)).iloc[0]
    assert fila["mode"] == "live" and fila["finished_at"] is not None
