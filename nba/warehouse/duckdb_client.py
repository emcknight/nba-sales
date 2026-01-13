from __future__ import annotations

import duckdb
from nba.config import settings


def connect():
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(settings.db_path))
