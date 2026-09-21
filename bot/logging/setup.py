"""Configuracion del logging estandar de Python con salida tambien a SQLite."""

from __future__ import annotations

import logging
import sys

from bot.config.schema import LoggingConfig
from bot.logging.db import Database


class SqliteErrorHandler(logging.Handler):
    """Vuelca WARNING y superiores a la tabla ``errors``.

    Un fallo al escribir en la base de datos nunca debe propagarse hacia el
    codigo que estaba logueando, asi que se traga la excepcion.
    """

    def __init__(self, db: Database, *, config_hash: str = "", run_id: int | None = None) -> None:
        super().__init__(level=logging.WARNING)
        self.db = db
        self.config_hash = config_hash
        self.run_id = run_id

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.db.log_error(
                source=record.name,
                message=record.getMessage(),
                level=record.levelname,
                exc=record.exc_info[1] if record.exc_info else None,
                config_hash=self.config_hash,
                run_id=self.run_id,
            )
        except Exception:  # noqa: BLE001 - el logging no puede tumbar al bot
            pass


def setup_logging(
    config: LoggingConfig,
    *,
    db: Database | None = None,
    config_hash: str = "",
    run_id: int | None = None,
) -> logging.Logger:
    """Deja el logger raiz listo; devuelve el logger del bot."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, config.level))
    for handler in list(root.handlers):
        root.removeHandler(handler)

    if config.console:
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)-28s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        root.addHandler(console)

    if db is not None:
        root.addHandler(SqliteErrorHandler(db, config_hash=config_hash, run_id=run_id))

    return logging.getLogger("bot")
