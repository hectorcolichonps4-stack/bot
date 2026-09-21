"""Test de integracion del bucle de operativa en modo paper."""

from __future__ import annotations

import pandas as pd
import pytest

from bot.data import DataFeed, ParquetCache, SyntheticProvider
from bot.execution import PaperBroker
from bot.logging import Database
from bot.runner import TradingSession


@pytest.fixture(autouse=True)
def sin_esperas(monkeypatch):
    """El bucle espera ``poll_seconds`` entre ciclos; en test no hay que esperar."""
    monkeypatch.setattr(TradingSession, "_sleep", staticmethod(lambda seconds: None))


@pytest.fixture
def session(loaded, tmp_path):
    proveedor = SyntheticProvider(bars=1200, seed=21, drift=0.0006, volatility=0.012)
    feed = DataFeed(proveedor, loaded.config, cache=ParquetCache(tmp_path / "cache"))
    db = Database(tmp_path / "bot.sqlite")
    run_id = db.start_run(mode="paper", config_hash=loaded.config_hash, config={})
    broker = PaperBroker(
        feed, initial_balance=10_000.0, config_hash=loaded.config_hash, db=db, run_id=run_id
    )
    sesion = TradingSession(loaded, feed=feed, broker=broker, db=db, run_id=run_id)
    yield sesion
    db.close()


def test_un_ciclo_completo_no_falla(session):
    decisiones = session.run_once()
    assert session.stats.iterations == 1
    assert len(decisiones) <= len(session.config.data.symbols)


def test_cada_ciclo_registra_la_curva_de_capital(session):
    session.run_once()
    session.run_once()
    assert len(session.db.equity_df(run_id=session.run_id)) == 2


def test_los_rechazos_llegan_a_la_base_de_datos(session):
    session.run_once()
    if session.stats.rejections == 0:
        pytest.skip("todas las senales pasaron los filtros en este ciclo")
    assert not session.db.rejections_df().empty


def test_el_kill_switch_detiene_el_bucle(session):
    session.risk.kill_switch.force("prueba", pd.Timestamp.now(tz="UTC"))
    session.run_forever(max_iterations=5)
    assert session.stats.iterations == 1  # sale tras el primer ciclo


def test_la_parada_ordenada_termina_el_bucle(session):
    session.request_stop()
    session.run_forever(max_iterations=10)
    assert session.stats.iterations == 0


def test_max_iterations_acota_el_bucle(session):
    session.run_forever(max_iterations=2)
    assert session.stats.iterations == 2


def test_un_fallo_de_datos_cuenta_como_error_sin_tumbar_la_sesion(session, monkeypatch):
    from bot.data.schema import DataError

    def explota(*args, **kwargs):
        raise DataError("proveedor caido")

    monkeypatch.setattr(session.feed, "latest", explota)
    session.run_once()
    assert session.stats.errors == len(session.config.data.symbols)


def test_la_sesion_guarda_su_resumen(session):
    session.request_stop()
    session.run_forever(max_iterations=1)
    assert session.db.get_state("last_session") is not None
