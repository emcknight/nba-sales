from __future__ import annotations

from pathlib import Path

import duckdb

from nba.config import settings


def connect(db_path: Path | None = None):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)

    path = db_path if db_path is not None else settings.db_path
    return duckdb.connect(str(path))
