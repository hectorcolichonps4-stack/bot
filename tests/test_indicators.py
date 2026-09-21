"""Tests de los indicadores.

Se comprueban valores conocidos a mano, los casos limite (series planas, solo
subidas, ventanas incompletas) y la ausencia de sesgo de anticipacion, que es
el error mas caro de todos: un indicador que mira al futuro convierte
cualquier backtest en ficcion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bot.indicators import (
    atr,
    atr_pct,
    ema,
    levels_from_pivots,
    macd,
    relative_volume,
    rsi,
    sma,
    support_resistance,
    swing_highs,
    swing_lows,
    true_range,
)


# ------------------------------------------------------------------- medias

def test_sma_valor_conocido():
    serie = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert sma(serie, 3).tolist()[2:] == [2.0, 3.0, 4.0]


def test_sma_deja_nan_hasta_completar_la_ventana():
    assert sma(pd.Series([1.0, 2.0, 3.0]), 3).isna().tolist() == [True, True, False]


def test_ema_sigue_la_formula_recursiva():
    serie = pd.Series([10.0, 20.0, 30.0, 40.0])
    resultado = ema(serie, 2)
    alpha = 2 / 3
    esperado = 15.0  # primera lectura: media de las dos primeras velas
    esperado = esperado + alpha * (30.0 - esperado)
    esperado = esperado + alpha * (40.0 - esperado)
    assert resultado.iloc[1] == pytest.approx(10.0 + alpha * (20.0 - 10.0))
    assert resultado.iloc[3] == pytest.approx(resultado.iloc[2] + alpha * (40.0 - resultado.iloc[2]))


def test_ema_de_serie_constante_es_esa_constante():
    serie = pd.Series([7.0] * 50)
    assert ema(serie, 10).dropna().unique().tolist() == [7.0]


def test_ema_rechaza_periodo_invalido():
    with pytest.raises(ValueError):
        ema(pd.Series([1.0, 2.0]), 0)


def test_ema_rechaza_entradas_no_numericas():
    with pytest.raises(TypeError):
        ema(pd.Series(["a", "b"]), 2)


# ---------------------------------------------------------------------- RSI

def test_rsi_de_serie_solo_alcista_es_100():
    serie = pd.Series(np.arange(1.0, 40.0))
    assert rsi(serie, 14).dropna().iloc[-1] == pytest.approx(100.0)


def test_rsi_de_serie_solo_bajista_es_0():
    serie = pd.Series(np.arange(40.0, 1.0, -1.0))
    assert rsi(serie, 14).dropna().iloc[-1] == pytest.approx(0.0)


def test_rsi_de_serie_plana_es_50():
    """Sin ganancias ni perdidas el RSI es indefinido; se fija en 50."""
    assert rsi(pd.Series([25.0] * 40), 14).dropna().iloc[-1] == pytest.approx(50.0)


def test_rsi_siempre_dentro_de_0_100(candles):
    valores = rsi(candles["close"], 14).dropna()
    assert valores.between(0.0, 100.0).all()


def test_rsi_necesita_calentamiento():
    assert rsi(pd.Series(np.arange(1.0, 20.0)), 14).iloc[:13].isna().all()


# --------------------------------------------------------------------- MACD

def test_macd_devuelve_las_tres_columnas(candles):
    resultado = macd(candles["close"])
    assert list(resultado.columns) == ["macd", "signal", "hist"]


def test_macd_hist_es_la_diferencia(candles):
    resultado = macd(candles["close"]).dropna()
    assert np.allclose(resultado["hist"], resultado["macd"] - resultado["signal"])


def test_macd_de_serie_constante_tiende_a_cero():
    resultado = macd(pd.Series([50.0] * 100)).dropna()
    assert resultado["macd"].abs().max() == pytest.approx(0.0, abs=1e-9)


def test_macd_exige_fast_menor_que_slow():
    with pytest.raises(ValueError):
        macd(pd.Series(np.arange(100.0)), fast=26, slow=12)


# ----------------------------------------------------------------- ATR y TR

def test_true_range_toma_el_mayor_de_los_tres_rangos():
    frame = pd.DataFrame({"high": [10.0, 12.0], "low": [8.0, 11.0], "close": [9.0, 11.5]})
    # Segunda vela: high-low = 1, |high-cierre previo| = 3, |low-cierre previo| = 2
    assert true_range(frame).tolist() == [2.0, 3.0]


def test_true_range_de_la_primera_vela_es_su_rango():
    frame = pd.DataFrame({"high": [10.0], "low": [8.0], "close": [9.0]})
    assert true_range(frame).iloc[0] == 2.0


def test_atr_es_positivo_y_suaviza(candles):
    valores = atr(candles, 14).dropna()
    assert (valores > 0).all()
    assert valores.std() < candles["close"].diff().abs().std() * 5


def test_atr_pct_es_relativo_al_precio(candles):
    esperado = atr(candles, 14) / candles["close"] * 100.0
    assert np.allclose(atr_pct(candles, 14).dropna(), esperado.dropna())


def test_atr_exige_columnas_ohlc():
    with pytest.raises(KeyError):
        atr(pd.DataFrame({"close": [1.0, 2.0]}), 14)


# ------------------------------------------------------------------ volumen

def test_volumen_relativo_compara_con_la_media_previa():
    volumen = pd.Series([100.0] * 20 + [300.0])
    assert relative_volume(volumen, 20).iloc[-1] == pytest.approx(3.0)


def test_volumen_relativo_excluye_la_vela_en_curso():
    """Incluirla diluiria justo el pico que se quiere detectar."""
    volumen = pd.Series([10.0] * 5 + [1000.0])
    resultado = relative_volume(volumen, 5).iloc[-1]
    assert resultado == pytest.approx(100.0)


def test_volumen_relativo_de_serie_constante_es_uno():
    assert relative_volume(pd.Series([42.0] * 30), 10).dropna().unique().tolist() == [1.0]


# ---------------------------------------------------- soportes/resistencias

def test_swing_high_detecta_el_pico():
    valores = [1.0, 2.0, 3.0, 9.0, 3.0, 2.0, 1.0]
    frame = pd.DataFrame({"high": valores, "low": valores, "close": valores})
    assert swing_highs(frame, 3).dropna().tolist() == [9.0]


def test_swing_low_detecta_el_valle():
    valores = [9.0, 8.0, 7.0, 1.0, 7.0, 8.0, 9.0]
    frame = pd.DataFrame({"high": valores, "low": valores, "close": valores})
    assert swing_lows(frame, 3).dropna().tolist() == [1.0]


def test_swings_no_miran_al_futuro():
    """Los ultimos ``window`` valores quedan sin confirmar, no adivinados."""
    valores = list(range(20))
    frame = pd.DataFrame({"high": valores, "low": valores, "close": valores}, dtype="float64")
    assert swing_highs(frame, 3).iloc[-3:].isna().all()


def test_agrupacion_funde_niveles_cercanos():
    niveles = levels_from_pivots([100.0, 100.2, 120.0], reference_price=110.0, tolerance_pct=1.0)
    assert niveles.supports == (pytest.approx(100.1),)
    assert niveles.resistances == (120.0,)


def test_niveles_se_reparten_alrededor_del_precio():
    niveles = levels_from_pivots([90.0, 95.0, 105.0, 110.0], reference_price=100.0, tolerance_pct=0.1)
    assert niveles.nearest_support == 95.0
    assert niveles.nearest_resistance == 105.0


def test_sin_pivotes_no_hay_niveles():
    niveles = levels_from_pivots([], reference_price=100.0)
    assert niveles.nearest_support is None and niveles.nearest_resistance is None


def test_support_resistance_sobre_velas_reales(candles):
    niveles = support_resistance(candles, lookback=120, pivot_window=3, tolerance_pct=0.4)
    precio = float(candles["close"].iloc[-1])
    assert all(nivel < precio for nivel in niveles.supports)
    assert all(nivel > precio for nivel in niveles.resistances)


def test_support_resistance_rechaza_lookback_insuficiente(candles):
    with pytest.raises(ValueError):
        support_resistance(candles, lookback=3, pivot_window=5)


# ----------------------------------------------------- ausencia de futuro

@pytest.mark.parametrize(
    "indicador",
    [
        lambda f: ema(f["close"], 20),
        lambda f: rsi(f["close"], 14),
        lambda f: macd(f["close"])["hist"],
        lambda f: atr(f, 14),
        lambda f: relative_volume(f["volume"], 20),
    ],
)
def test_los_indicadores_no_usan_velas_futuras(candles, indicador):
    """El valor en la vela N debe ser el mismo con o sin las velas posteriores."""
    corte = 300
    completo = indicador(candles).iloc[corte]
    truncado = indicador(candles.iloc[: corte + 1]).iloc[corte]
    assert completo == pytest.approx(truncado)
