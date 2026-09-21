"""Registro persistente en SQLite y configuracion del logging.

El paquete se llama ``bot.logging`` y nunca ensombrece al ``logging`` de la
libreria estandar porque siempre vive dentro del paquete ``bot``.
"""

from bot.logging.db import SCHEMA_VERSION, Database
from bot.logging.setup import SqliteErrorHandler, setup_logging

__all__ = ["SCHEMA_VERSION", "Database", "SqliteErrorHandler", "setup_logging"]
