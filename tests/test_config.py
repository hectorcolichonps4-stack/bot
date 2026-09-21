"""Tests de configuracion: validacion estricta y hash reproducible."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bot.config import DEFAULT_CONFIG_PATH, ConfigError, compute_config_hash, load_config
from bot.config.schema import Config, GatesConfig, IndicatorsConfig, MacdConfig, RiskConfig


def test_el_settings_de_ejemplo_es_valido():
    cargada = load_config(DEFAULT_CONFIG_PATH)
    assert cargada.config.data.symbols
    assert len(cargada.config_hash) == 16


def test_una_clave_desconocida_es_un_error():
    """Un typo debe romper el arranque, no ignorarse en silencio."""
    with pytest.raises(ValidationError):
        Config.model_validate({"risk": {"risk_per_trade_ptc": 1.0}})


def test_las_emas_deben_ir_de_menor_a_mayor():
    with pytest.raises(ValidationError):
        IndicatorsConfig(ema_fast=50, ema_slow=20, ema_trend=200)


def test_macd_exige_fast_menor_que_slow():
    with pytest.raises(ValidationError):
        MacdConfig(fast=26, slow=12)


def test_el_atr_minimo_debe_ser_menor_que_el_maximo():
    with pytest.raises(ValidationError):
        GatesConfig(min_atr_pct=9.0, max_atr_pct=2.0)


def test_el_limite_diario_no_puede_superar_al_semanal():
    with pytest.raises(ValidationError):
        RiskConfig(max_daily_loss_pct=10.0, max_weekly_loss_pct=5.0)


def test_el_calentamiento_debe_cubrir_los_indicadores():
    """Con warmup corto los indicadores nunca se llenan y el bot no opera."""
    with pytest.raises(ValidationError):
        Config.model_validate({"data": {"warmup_bars": 100}})


def test_los_simbolos_duplicados_se_rechazan():
    with pytest.raises(ValidationError):
        Config.model_validate({"data": {"symbols": ["BTCUSDT", "BTCUSDT"]}})


def test_los_pesos_no_admiten_valores_negativos():
    with pytest.raises(ValidationError):
        Config.model_validate({"scoring": {"weights": {"trend": -1.0}}})


def test_la_configuracion_es_inmutable():
    config = Config()
    with pytest.raises(ValidationError):
        config.risk.risk_per_trade_pct = 5.0


def test_el_hash_es_estable_entre_instancias_equivalentes():
    assert compute_config_hash(Config()) == compute_config_hash(Config())


def test_el_hash_cambia_al_cambiar_un_parametro():
    otra = Config.model_validate({"risk": {"risk_per_trade_pct": 2.0}})
    assert compute_config_hash(Config()) != compute_config_hash(otra)


def test_el_hash_ignora_los_valores_por_defecto_omitidos():
    """Escribir un valor que ya era el de por defecto no cambia la huella."""
    explicita = Config.model_validate({"risk": {"risk_per_trade_pct": RiskConfig().risk_per_trade_pct}})
    assert compute_config_hash(Config()) == compute_config_hash(explicita)


def test_los_overrides_se_fusionan_en_profundidad():
    cargada = load_config(DEFAULT_CONFIG_PATH, overrides={"risk": {"max_daily_trades": 3}})
    assert cargada.config.risk.max_daily_trades == 3
    assert cargada.config.risk.risk_per_trade_pct == RiskConfig().risk_per_trade_pct


def test_un_fichero_inexistente_da_error_claro():
    with pytest.raises(ConfigError, match="no existe"):
        load_config("/no/existe/settings.yaml")


def test_las_variables_de_entorno_se_expanden(tmp_path, monkeypatch):
    monkeypatch.setenv("BOT_TEST_TF", "1h")
    ruta = tmp_path / "settings.yaml"
    ruta.write_text("data:\n  timeframe: ${BOT_TEST_TF}\n  warmup_bars: 400\n", encoding="utf-8")
    assert load_config(ruta).config.data.timeframe == "1h"


def test_una_variable_de_entorno_admite_valor_por_defecto(tmp_path):
    ruta = tmp_path / "settings.yaml"
    ruta.write_text("data:\n  timeframe: ${BOT_SIN_DEFINIR:-4h}\n  warmup_bars: 400\n", encoding="utf-8")
    assert load_config(ruta).config.data.timeframe == "4h"


def test_una_variable_sin_definir_ni_defecto_falla(tmp_path):
    ruta = tmp_path / "settings.yaml"
    ruta.write_text("data:\n  timeframe: ${BOT_NO_EXISTE_SEGURO}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="variable de entorno"):
        load_config(ruta)


def test_un_yaml_invalido_da_error_de_configuracion(tmp_path):
    ruta = tmp_path / "settings.yaml"
    ruta.write_text("data: [esto\n  no: cierra\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(ruta)
