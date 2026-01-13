from pathlib import Path

from nba.pipelines.generate_synthetic import SalesSynthConfig, run
from nba.warehouse.duckdb_client import connect


def test_generate_and_load_creates_tables(tmp_path):
    db_path = Path(tmp_path) / "warehouse.duckdb"

    cfg = SalesSynthConfig(n_accounts=300, n_days_history=60)
    counts = run(cfg, db_path=db_path)

    assert counts["raw_accounts"] == 300
    assert counts["raw_signals_daily"] > 0
    assert counts["raw_outcomes"] == 300

    with connect(db_path=db_path) as con:
        tables = set(con.execute("SHOW TABLES").fetchdf()["name"].tolist())
        assert "raw_accounts" in tables
        assert "raw_signals_daily" in tables
        assert "raw_touches" in tables
        assert "raw_actions" in tables
        assert "raw_outcomes" in tables
