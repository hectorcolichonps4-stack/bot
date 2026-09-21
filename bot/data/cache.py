"""Cache de velas en parquet.

Un fichero por (simbolo, timeframe). Las escrituras son atomicas (fichero
temporal + rename) para que una interrupcion no deje un parquet a medias.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from bot.data.schema import empty_frame, normalize


class ParquetCache:
    """Almacen local de velas historicas."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, symbol: str, timeframe: str) -> Path:
        return self.root / f"{symbol.upper()}_{timeframe}.parquet"

    def load(self, symbol: str, timeframe: str) -> pd.DataFrame:
        path = self.path_for(symbol, timeframe)
        if not path.exists():
            return empty_frame()
        try:
            return normalize(pd.read_parquet(path))
        except Exception:
            # Un parquet corrupto no debe tumbar el bot: se descarta y se
            # vuelve a descargar desde el exchange.
            path.unlink(missing_ok=True)
            return empty_frame()

    def store(self, symbol: str, timeframe: str, frame: pd.DataFrame) -> Path:
        path = self.path_for(symbol, timeframe)
        tmp = path.with_suffix(".parquet.tmp")
        normalize(frame).to_parquet(tmp, index=False)
        os.replace(tmp, path)
        return path

    def merge(self, symbol: str, timeframe: str, frame: pd.DataFrame) -> pd.DataFrame:
        """Fusiona velas nuevas con las cacheadas y persiste el resultado.

        Las velas nuevas ganan ante un mismo ``open_time``: la ultima vela de
        una descarga anterior pudo guardarse sin cerrar.
        """
        existing = self.load(symbol, timeframe)
        combined = normalize(pd.concat([existing, frame], ignore_index=True))
        self.store(symbol, timeframe, combined)
        return combined

    def coverage(self, symbol: str, timeframe: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        frame = self.load(symbol, timeframe)
        if frame.empty:
            return None, None
        return frame["open_time"].iloc[0], frame["open_time"].iloc[-1]

    def clear(self, symbol: str | None = None, timeframe: str | None = None) -> int:
        pattern = f"{symbol.upper() if symbol else '*'}_{timeframe or '*'}.parquet"
        removed = 0
        for path in self.root.glob(pattern):
            path.unlink()
            removed += 1
        return removed
