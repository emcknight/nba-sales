from __future__ import annotations

import duckdb
import pandas as pd
from nba.warehouse.duckdb_client import connect


def build_training_set(db_path=None) -> pd.DataFrame:
    with connect(db_path=db_path) as con:
        # Load tables
        accounts = con.execute("SELECT * FROM raw_accounts").fetchdf()
        feats = con.execute("SELECT * FROM raw_signals_daily").fetchdf()
        touches = con.execute("SELECT * FROM raw_touches").fetchdf()
        actions = con.execute("SELECT * FROM raw_actions").fetchdf()
        outcomes = con.execute("SELECT * FROM raw_outcomes").fetchdf()

        # --- Aggregate signals (trailing 30d) ---
        feats["ds"] = pd.to_datetime(feats["ds"])
        as_of = pd.to_datetime(actions["as_of_date"].iloc[0])
        recent_mask = feats["ds"] >= as_of - pd.Timedelta(days=30)
        agg = (
            feats.loc[recent_mask]
            .groupby("account_id")
            .agg(
                web_30=("web_visits", "sum"),
                downloads_30=("content_downloads", "sum"),
                trial_users_30=("trial_active_users", "mean"),
                trial_events_30=("trial_events", "sum"),
            )
            .reset_index()
        )

        # --- Aggregate touches ---
        touches["touch_date"] = pd.to_datetime(touches["touch_date"], errors="coerce")
        tmask = touches["touch_date"] >= as_of - pd.Timedelta(days=30)
        t_agg = (
            touches.loc[tmask]
            .groupby("account_id")
            .agg(
                touches_30=("touch_type", "count"),
                responses_30=("responded", "sum"),
            )
            .reset_index()
        )

        # --- Join everything ---
        df = (
            accounts.merge(agg, on="account_id", how="left")
            .merge(t_agg, on="account_id", how="left")
            .merge(actions, on="account_id", how="left")
            .merge(outcomes, on=["account_id", "as_of_date"], how="left")
        )

        df = df.fillna(0)

        # Create target & weights
        df["y_won"] = df["won"].astype(int)
        df["y_revenue"] = df["realized_revenue"].astype(float)

        # Optional: simple cost assumptions for each action
        cost_map = {
            "EMAIL_SEQUENCE": 5,
            "CALL_OUTREACH": 25,
            "LINKEDIN_TOUCH": 10,
            "DEMO_OFFER": 50,
            "EXEC_SPONSOR_OUTREACH": 200,
            "PRICING_CONCESSION": 100,
            "TECHNICAL_WORKSHOP": 300,
            None: 0,
        }
        df["action_cost"] = df["action_type"].map(cost_map).fillna(0)

        # ensure all *_date columns are datetime
        for col in df.columns:
            if "date" in col:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        con.execute("CREATE OR REPLACE TABLE train_sales_nba AS SELECT * FROM df")

        return df
