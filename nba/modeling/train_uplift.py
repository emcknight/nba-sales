from __future__ import annotations

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from nba.warehouse.duckdb_client import connect
from nba.config import settings


def train_uplift(db_path=None):
    """Train per-action T-learners to estimate incremental win probability."""
    with connect(db_path=db_path) as con:
        df = con.execute("SELECT * FROM train_sales_nba").fetchdf()

    df = df.fillna(0)

    base_features = [
        "web_30", "downloads_30", "trial_users_30", "trial_events_30",
        "touches_30", "responses_30", "intent_affinity", "employees",
    ]

    actions = df["action_type"].dropna().unique().tolist()
    results = []

    for act in actions:
        df_act = df.copy()
        df_act["treat"] = (df_act["action_type"] == act).astype(int)

        X = df_act[base_features + ["treat"]]
        y = df_act["y_won"].astype(int)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        model = GradientBoostingClassifier(random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        print(f"✅ {act}: AUC={auc:.3f}")

        # Save model
        artifact = settings.artifacts_dir / f"uplift_{act}.pkl"
        joblib.dump(model, artifact)

        results.append((act, auc))

    print("\nAll models trained.")
    print(pd.DataFrame(results, columns=["action", "auc"]))

    return results
