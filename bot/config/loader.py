"""Carga, validacion y hashing de la configuracion.

El hash es la pieza clave para la trazabilidad: cada trade y cada senal se
guardan junto al hash de la configuracion exacta con la que se generaron, de
modo que un resultado historico siempre se puede reatribuir a sus parametros.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import ValidationError

from bot.config.schema import Config

DEFAULT_CONFIG_PATH = Path(__file__).with_name("settings.yaml")

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigError(RuntimeError):
    """Configuracion ausente, mal formada o invalida."""


@dataclass(frozen=True)
class LoadedConfig:
    """Configuracion validada junto con su procedencia y su huella."""

    config: Config
    config_hash: str
    source_path: Path | None
    raw: Mapping[str, Any]

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - azucar sintactico
        return getattr(self.config, item)


def _expand_env(value: Any) -> Any:
    """Sustituye ``${VAR}`` y ``${VAR:-default}`` en las cadenas del YAML."""
    if isinstance(value, str):
        def repl(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            resolved = os.environ.get(name)
            if resolved is None:
                if default is None:
                    raise ConfigError(f"variable de entorno no definida: {name}")
                return default
            return resolved

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _canonical(payload: Any) -> Any:
    """Normaliza el payload para que el hash sea estable entre ejecuciones."""
    if isinstance(payload, dict):
        return {k: _canonical(payload[k]) for k in sorted(payload)}
    if isinstance(payload, (list, tuple)):
        return [_canonical(v) for v in payload]
    return payload


def compute_config_hash(config: Config) -> str:
    """Huella corta y determinista del contenido efectivo de la configuracion.

    Se calcula sobre el modelo ya validado (no sobre el texto del YAML), por lo
    que comentarios, orden de claves y valores por defecto omitidos no alteran
    el resultado: dos configuraciones equivalentes comparten hash.
    """
    payload = config.model_dump(mode="json")
    blob = json.dumps(_canonical(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def load_config(path: str | os.PathLike[str] | None = None, *, overrides: Mapping[str, Any] | None = None) -> LoadedConfig:
    """Lee el YAML, aplica overrides, valida y devuelve la configuracion sellada."""
    source = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not source.exists():
        raise ConfigError(f"no existe el fichero de configuracion: {source}")

    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML invalido en {source}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"la raiz de {source} debe ser un mapa de claves")

    raw = _expand_env(raw)
    if overrides:
        raw = _deep_merge(raw, overrides)

    try:
        config = Config.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"configuracion invalida en {source}:\n{exc}") from exc

    return LoadedConfig(
        config=config,
        config_hash=compute_config_hash(config),
        source_path=source,
        raw=raw,
    )


def _deep_merge(base: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
