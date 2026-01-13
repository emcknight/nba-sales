from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = project_root / "data"
    db_path: Path = data_dir / "warehouse.duckdb"
    raw_dir: Path = data_dir / "raw"
    artifacts_dir: Path = data_dir / "artifacts"


settings = Settings()
