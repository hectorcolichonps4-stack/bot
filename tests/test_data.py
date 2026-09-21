"""Tests de la capa de datos: normalizacion, validacion y cache."""

from __future__ import annotations

import pandas as pd
import pytest

from bot.data import (
    DataError,
    DataFeed,
    ParquetCache,
    SyntheticProvider,
    drop_unclosed,
    empty_frame,
    normalize,
    timeframe_to_timedelta,
    validate,
)


def test_el_frame_vacio_tiene_el_esquema_correcto():
    frame = empty_frame()
    assert "open_time" in frame.columns and frame.empty


def test_normalize_ordena_y_deduplica():
    base = SyntheticProvider(bars=10).fetch_klines("BTCUSDT", "15m")
    revuelto = pd.concat([base.iloc[::-1], base.iloc[:3]], ignore_index=True)
    resultado = normalize(revuelto)
    assert len(resultado) == 10
    assert resultado["open_time"].is_monotonic_increasing


def test_normalize_exige_todas_las_columnas():
    with pytest.raises(DataError, match="faltan columnas"):
        normalize(pd.DataFrame({"open_time": [1], "close": [2]}))


def test_validate_detecta_high_menor_que_low():
    frame = SyntheticProvider(bars=10).fetch_klines("BTCUSDT", "15m")
    frame.loc[5, "high"] = frame.loc[5, "low"] - 1.0
    with pytest.raises(DataError, match="high < low"):
        validate(frame)


def test_validate_detecta_cierres_fuera_de_rango():
    frame = SyntheticProvider(bars=10).fetch_klines("BTCUSDT", "15m")
    frame.loc[5, "close"] = frame.loc[5, "high"] + 10.0
    with pytest.raises(DataError, match="fuera del rango"):
        validate(frame)


def test_validate_detecta_volumen_negativo():
    frame = SyntheticProvider(bars=10).fetch_klines("BTCUSDT", "15m")
    frame.loc[5, "volume"] = -1.0
    with pytest.raises(DataError, match="negativo"):
        validate(frame)


def test_validate_anota_los_huecos_sin_fallar():
    """Los exchanges tienen paradas: es un aviso, no un motivo para abortar."""
    frame = SyntheticProvider(bars=20).fetch_klines("BTCUSDT", "15m")
    frame = frame.drop(index=[5, 6]).reset_index(drop=True)
    resultado = validate(frame, timeframe="15m")
    assert resultado.attrs.get("gaps", 0) >= 1


def test_timeframe_desconocido_falla():
    with pytest.raises(DataError, match="no soportado"):
        timeframe_to_timedelta("7m")


def test_drop_unclosed_descarta_la_vela_en_curso():
    ahora = pd.Timestamp.now(tz="UTC")
    frame = SyntheticProvider(bars=5, anchor=ahora.ceil("min")).fetch_klines("BTCUSDT", "15m")
    assert len(drop_unclosed(frame, "15m")) == len(frame) - 1


# -------------------------------------------------------------------- cache

def test_la_cache_persiste_y_recupera(tmp_path):
    cache = ParquetCache(tmp_path)
    frame = SyntheticProvider(bars=50).fetch_klines("BTCUSDT", "15m")
    cache.store("BTCUSDT", "15m", frame)
    assert len(cache.load("BTCUSDT", "15m")) == 50


def test_la_cache_de_un_simbolo_desconocido_esta_vacia(tmp_path):
    assert ParquetCache(tmp_path).load("NOEXISTE", "15m").empty


def test_merge_no_duplica_velas(tmp_path):
    cache = ParquetCache(tmp_path)
    frame = SyntheticProvider(bars=50).fetch_klines("BTCUSDT", "15m")
    cache.merge("BTCUSDT", "15m", frame)
    resultado = cache.merge("BTCUSDT", "15m", frame)
    assert len(resultado) == 50


def test_merge_deja_ganar_a_las_velas_nuevas(tmp_path):
    """La ultima vela cacheada pudo guardarse sin cerrar."""
    cache = ParquetCache(tmp_path)
    frame = SyntheticProvider(bars=10).fetch_klines("BTCUSDT", "15m")
    cache.store("BTCUSDT", "15m", frame)

    corregida = frame.tail(1).copy()
    corregida.loc[corregida.index[0], "close"] = 12345.0
    resultado = cache.merge("BTCUSDT", "15m", corregida)
    assert resultado["close"].iloc[-1] == 12345.0


def test_un_parquet_corrupto_se_descarta(tmp_path):
    cache = ParquetCache(tmp_path)
    cache.path_for("BTCUSDT", "15m").write_text("esto no es parquet", encoding="utf-8")
    assert cache.load("BTCUSDT", "15m").empty
    assert not cache.path_for("BTCUSDT", "15m").exists()


def test_coverage_informa_del_rango(tmp_path):
    cache = ParquetCache(tmp_path)
    frame = SyntheticProvider(bars=30).fetch_klines("BTCUSDT", "15m")
    cache.store("BTCUSDT", "15m", frame)
    primera, ultima = cache.coverage("BTCUSDT", "15m")
    assert primera == frame["open_time"].iloc[0] and ultima == frame["open_time"].iloc[-1]


# --------------------------------------------------------------------- feed

def test_el_feed_usa_la_cache_en_la_segunda_pasada(config, tmp_path):
    feed = DataFeed(SyntheticProvider(bars=600), config, cache=ParquetCache(tmp_path))
    feed.history("BTCUSDT")
    assert feed.stats["BTCUSDT"].from_cache == 0
    feed.history("BTCUSDT")
    assert feed.stats["BTCUSDT"].from_cache > 0


def test_el_feed_sin_refresco_no_llama_al_proveedor(config, tmp_path):
    class Explosivo(SyntheticProvider):
        def fetch_klines(self, *args, **kwargs):  # pragma: no cover - no deberia llamarse
            raise AssertionError("no deberia descargar con refresh=False")

    cache = ParquetCache(tmp_path)
    cache.store("BTCUSDT", config.data.timeframe, SyntheticProvider(bars=50).fetch_klines("BTCUSDT", "15m"))
    feed = DataFeed(Explosivo(), config, cache=cache)
    assert len(feed.history("BTCUSDT", refresh=False)) == 50


def test_bulk_history_ignora_los_simbolos_caidos(config, tmp_path):
    class Parcial(SyntheticProvider):
        def fetch_klines(self, symbol, *args, **kwargs):
            if symbol == "ETHUSDT":
                raise DataError("caido")
            return super().fetch_klines(symbol, *args, **kwargs)

    feed = DataFeed(Parcial(bars=400), config, cache=ParquetCache(tmp_path))
    resultado = feed.bulk_history(["BTCUSDT", "ETHUSDT"])
    assert set(resultado) == {"BTCUSDT"}


def test_bulk_history_falla_si_no_queda_ningun_simbolo(config, tmp_path):
    class Caido(SyntheticProvider):
        def fetch_klines(self, *args, **kwargs):
            raise DataError("caido")

    feed = DataFeed(Caido(), config, cache=ParquetCache(tmp_path))
    with pytest.raises(DataError, match="ningun simbolo"):
        feed.bulk_history(["BTCUSDT"])


def test_los_filtros_caen_a_permisivos_si_el_exchange_falla(config, tmp_path):
    class SinFiltros(SyntheticProvider):
        def fetch_symbol_filters(self, symbol):
            raise DataError("sin exchangeInfo")

    feed = DataFeed(SinFiltros(bars=100), config, cache=ParquetCache(tmp_path))
    assert feed.filters("BTCUSDT").symbol == "BTCUSDT"
