from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from nba.warehouse.duckdb_client import connect


@dataclass(frozen=True)
class SalesSynthConfig:
    seed: int = 42
    n_accounts: int = 5000

    # History for features before as_of_date
    n_days_history: int = 180

    # Outcome horizon after as_of_date (e.g., close within 90 days)
    horizon_days: int = 90

    # Snapshot date
    as_of_date: date = date(2026, 1, 1)

    # Share of accounts receiving any action near as_of_date
    action_rate: float = 0.22


ACTIONS = [
    "EMAIL_SEQUENCE",
    "CALL_OUTREACH",
    "LINKEDIN_TOUCH",
    "DEMO_OFFER",
    "EXEC_SPONSOR_OUTREACH",
    "PRICING_CONCESSION",
    "TECHNICAL_WORKSHOP",
]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _make_accounts(rng: np.random.Generator, cfg: SalesSynthConfig) -> pd.DataFrame:
    industries = np.array(["SaaS", "Ecom", "FinTech", "Health", "EdTech", "Manufacturing", "Media"])
    segments = np.array(["SMB", "MM", "ENT"])
    geo = np.array(["NA", "EMEA", "APAC"])

    account_id = np.arange(1, cfg.n_accounts + 1)
    segment = rng.choice(segments, size=cfg.n_accounts, p=[0.62, 0.26, 0.12])
    industry = rng.choice(
        industries, size=cfg.n_accounts, p=[0.24, 0.15, 0.14, 0.10, 0.09, 0.13, 0.15]
    )
    region = rng.choice(geo, size=cfg.n_accounts, p=[0.55, 0.30, 0.15])

    # Firmographics / sizing
    employees = np.where(
        segment == "SMB",
        rng.integers(20, 250, cfg.n_accounts),
        np.where(
            segment == "MM",
            rng.integers(250, 2000, cfg.n_accounts),
            rng.integers(2000, 25000, cfg.n_accounts),
        ),
    ).astype(int)

    # Potential ACV proxy (higher for ENT, higher for some industries)
    ind_mult = (
        pd.Series(industry)
        .map(
            {
                "SaaS": 1.10,
                "Ecom": 0.95,
                "FinTech": 1.05,
                "Health": 1.00,
                "EdTech": 0.85,
                "Manufacturing": 0.98,
                "Media": 0.90,
            }
        )
        .to_numpy()
    )
    seg_mult = pd.Series(segment).map({"SMB": 0.6, "MM": 1.0, "ENT": 1.7}).to_numpy()

    base_acv = (employees * rng.uniform(35, 90, cfg.n_accounts) * ind_mult * seg_mult).round(0)
    # Keep ACV in a plausible range
    acv_potential = np.clip(base_acv, 3000, 750000).astype(int)

    # Latent “intent affinity” (how likely they are to show buying intent)
    # Drives inbound signals and responsiveness
    ind_intent = (
        pd.Series(industry)
        .map(
            {
                "SaaS": 0.10,
                "Ecom": 0.02,
                "FinTech": 0.05,
                "Health": 0.00,
                "EdTech": -0.05,
                "Manufacturing": -0.02,
                "Media": -0.03,
            }
        )
        .to_numpy()
    )
    seg_intent = pd.Series(segment).map({"SMB": -0.02, "MM": 0.03, "ENT": 0.08}).to_numpy()
    intent_affinity = np.clip(
        0.45 + ind_intent + seg_intent + rng.normal(0, 0.12, cfg.n_accounts), 0.05, 0.95
    )

    # Created date
    created_days_ago = rng.integers(30, 1800, cfg.n_accounts)
    created_date = pd.to_datetime(cfg.as_of_date) - pd.to_timedelta(created_days_ago, unit="D")

    return pd.DataFrame(
        {
            "account_id": account_id,
            "segment": segment,
            "industry": industry,
            "region": region,
            "employees": employees,
            "acv_potential": acv_potential,
            "intent_affinity": intent_affinity.astype(float),
            "created_date": created_date.date,
        }
    )


def _make_signals_daily(
    rng: np.random.Generator, cfg: SalesSynthConfig, accounts: pd.DataFrame
) -> pd.DataFrame:
    """
    Daily “signals” that sales might use:
      - web_visits (intent)
      - content_downloads (intent)
      - trial_events (PLG usage proxy)
      - trial_active_users
    """
    start = cfg.as_of_date - timedelta(days=cfg.n_days_history)
    days = pd.date_range(start=start, periods=cfg.n_days_history, freq="D")

    affinity = accounts["intent_affinity"].to_numpy()
    seg = accounts["segment"].to_numpy()
    emp = accounts["employees"].to_numpy()

    # Segment multipliers (enterprise tends to have more stakeholders, more touches, but slower)
    seg_mult = np.where(seg == "SMB", 1.0, np.where(seg == "MM", 1.15, 1.25))

    # Trial adoption propensity (PLG-ish)
    trial_prop = np.clip(0.15 + 0.55 * affinity + rng.normal(0, 0.08, len(accounts)), 0.02, 0.95)

    rows = []
    for i, d in enumerate(days):
        dow = d.dayofweek
        week_factor = 0.85 if dow >= 5 else 1.0

        # Some mild time dynamics: slight upward drift in intent for some accounts
        drift = 1.0 + (i / cfg.n_days_history) * rng.normal(0.02, 0.02, len(accounts))
        drift = np.clip(drift, 0.85, 1.20)

        # Web visits: driven by affinity, plus noise; enterprise gets slightly more
        lam_visits = np.clip((0.6 + 3.2 * affinity) * seg_mult * week_factor * drift, 0.05, None)
        web_visits = rng.poisson(lam=lam_visits).astype(int)

        # Content downloads: rarer, increases with web visits & affinity
        lam_dl = np.clip(
            (0.03 + 0.06 * affinity) * (1.0 + 0.12 * web_visits) * week_factor, 0.001, None
        )
        content_downloads = rng.poisson(lam=lam_dl).astype(int)

        # Trial events: only for accounts with trials; events roughly scale with employees (capped)
        has_trial = rng.uniform(size=len(accounts)) < trial_prop
        trial_active_users = np.where(
            has_trial,
            rng.poisson(
                lam=np.clip(
                    (0.5 + 2.0 * affinity) * np.sqrt(np.clip(emp, 20, 500) / 50.0), 0.2, 25.0
                )
            ),
            0,
        ).astype(int)
        trial_events = np.where(
            has_trial,
            rng.poisson(lam=np.clip(trial_active_users * rng.uniform(3.0, 7.0), 0.2, 180.0)),
            0,
        ).astype(int)

        rows.append(
            pd.DataFrame(
                {
                    "account_id": accounts["account_id"].to_numpy(),
                    "ds": pd.to_datetime(d).date(),
                    "web_visits": web_visits,
                    "content_downloads": content_downloads,
                    "trial_active_users": trial_active_users,
                    "trial_events": trial_events,
                }
            )
        )

    return pd.concat(rows, ignore_index=True)


def _make_touches(
    rng: np.random.Generator,
    cfg: SalesSynthConfig,
    accounts: pd.DataFrame,
    signals_daily: pd.DataFrame,
) -> pd.DataFrame:
    """
    Touches are logged activities before as_of_date:
      - emails, calls, linkedin
    More touches happen for higher intent and for larger segments.
    """
    df = signals_daily.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    as_of = pd.to_datetime(cfg.as_of_date)

    # Use trailing 30d intent to drive touches
    window_30 = (df["ds"] >= as_of - pd.Timedelta(days=30)) & (df["ds"] < as_of)
    agg = (
        df.loc[window_30]
        .groupby("account_id")
        .agg(
            web_30=("web_visits", "sum"),
            dl_30=("content_downloads", "sum"),
            trial_events_30=("trial_events", "sum"),
            tau_30=("trial_active_users", "mean"),
        )
        .reset_index()
    )

    a = accounts.merge(agg, on="account_id", how="left").fillna(0)

    seg = a["segment"].to_numpy()
    seg_mult = np.where(seg == "SMB", 1.0, np.where(seg == "MM", 1.2, 1.35))

    # Touch intensity as a function of intent
    intent_score = sigmoid(
        -1.2
        + 0.015 * np.log1p(a["web_30"].to_numpy())
        + 0.08 * np.log1p(a["dl_30"].to_numpy())
        + 0.01 * np.log1p(a["trial_events_30"].to_numpy())
        + 0.05 * np.log1p(a["tau_30"].to_numpy() + 1.0)
        + 0.15 * (a["segment"].eq("ENT")).astype(float).to_numpy()
    )

    # Number of touches in last 30 days
    lam_touches = np.clip((0.6 + 6.0 * intent_score) * seg_mult, 0.2, 18.0)
    n_touches = rng.poisson(lam=lam_touches).astype(int)

    touch_types = np.array(["EMAIL", "CALL", "LINKEDIN"])
    # Email dominates, calls increase with intent and segment
    rows = []
    for account_id, nt, is_ent, score in zip(
        a["account_id"].to_numpy(), n_touches, a["segment"].eq("ENT"), intent_score
    ):
        if nt == 0:
            continue

        p_call = np.clip(0.12 + 0.20 * score + (0.08 if is_ent else 0.0), 0.10, 0.55)
        p_li = np.clip(0.10 + 0.10 * score, 0.08, 0.25)
        p_email = max(1.0 - p_call - p_li, 0.20)
        p = np.array([p_email, p_call, p_li])
        p = p / p.sum()

        # Touch dates spread over last 30 days
        days_ago = rng.integers(1, 31, size=nt)
        touch_date = (pd.to_datetime(cfg.as_of_date) - pd.to_timedelta(days_ago, unit="D")).date

        types = rng.choice(touch_types, size=nt, p=p)
        # Simple response probability (more likely if high intent)
        responded = (rng.uniform(size=nt) < np.clip(0.05 + 0.25 * score, 0.05, 0.35)).astype(int)

        rows.append(
            pd.DataFrame(
                {
                    "account_id": np.full(nt, account_id, dtype=int),
                    "touch_date": touch_date,
                    "touch_type": types,
                    "responded": responded,
                }
            )
        )

    return (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["account_id", "touch_date", "touch_type", "responded"])
    )


def _make_features_as_of(
    cfg: SalesSynthConfig, signals_daily: pd.DataFrame, touches: pd.DataFrame
) -> pd.DataFrame:
    s = signals_daily.copy()
    s["ds"] = pd.to_datetime(s["ds"])
    as_of = pd.to_datetime(cfg.as_of_date)

    w30 = (s["ds"] >= as_of - pd.Timedelta(days=30)) & (s["ds"] < as_of)
    w90 = (s["ds"] >= as_of - pd.Timedelta(days=90)) & (s["ds"] < as_of)

    g30 = (
        s.loc[w30]
        .groupby("account_id")
        .agg(
            web_30=("web_visits", "sum"),
            dl_30=("content_downloads", "sum"),
            tau_30=("trial_active_users", "mean"),
            trial_events_30=("trial_events", "sum"),
            active_days_30=("trial_active_users", lambda x: int((x > 0).sum())),
        )
    )
    g90 = (
        s.loc[w90]
        .groupby("account_id")
        .agg(
            web_90=("web_visits", "sum"),
            dl_90=("content_downloads", "sum"),
            tau_90=("trial_active_users", "mean"),
            trial_events_90=("trial_events", "sum"),
        )
    )

    feats = g30.join(g90, how="left").reset_index().fillna(0)
    feats["as_of_date"] = cfg.as_of_date

    # Touch features in trailing 30d
    if len(touches) > 0:
        t = touches.copy()
        t["touch_date"] = pd.to_datetime(t["touch_date"])
        t30 = t[(t["touch_date"] >= as_of - pd.Timedelta(days=30)) & (t["touch_date"] < as_of)]
        tg = (
            t30.groupby("account_id")
            .agg(
                touches_30=("touch_type", "count"),
                calls_30=("touch_type", lambda x: int((x == "CALL").sum())),
                emails_30=("touch_type", lambda x: int((x == "EMAIL").sum())),
                linkedin_30=("touch_type", lambda x: int((x == "LINKEDIN").sum())),
                responses_30=("responded", "sum"),
            )
            .reset_index()
        )
        feats = feats.merge(tg, on="account_id", how="left").fillna(0)
    else:
        feats["touches_30"] = 0
        feats["calls_30"] = 0
        feats["emails_30"] = 0
        feats["linkedin_30"] = 0
        feats["responses_30"] = 0

    return feats


def _assign_actions(
    rng: np.random.Generator,
    cfg: SalesSynthConfig,
    accounts: pd.DataFrame,
    feats: pd.DataFrame,
) -> pd.DataFrame:
    df = accounts.merge(feats, on="account_id", how="left")

    # Sales “priority” score: high intent + trial activity + recent responses
    intent_score = sigmoid(
        -1.1
        + 0.012 * np.log1p(df["web_30"].to_numpy())
        + 0.10 * np.log1p(df["dl_30"].to_numpy())
        + 0.012 * np.log1p(df["trial_events_30"].to_numpy())
        + 0.09 * np.log1p(df["responses_30"].to_numpy() + 1.0)
    )

    # Selection bias: reps disproportionately act on higher intent accounts,
    # and enterprise gets more attention.
    ent_boost = (df["segment"] == "ENT").astype(float) * 0.18
    propensity_any = sigmoid(-2.0 + 3.4 * intent_score + ent_boost)

    # Calibrate to target action_rate
    scale = cfg.action_rate / max(propensity_any.mean(), 1e-6)
    propensity_any = np.clip(propensity_any * scale, 0.0, 0.97)

    treated = rng.uniform(size=len(df)) < propensity_any

    # Action mix depends on segment + signals
    probs = np.zeros((len(df), len(ACTIONS)), dtype=float)

    # Base distribution (cheap actions more common)
    probs[:, ACTIONS.index("EMAIL_SEQUENCE")] = 0.30
    probs[:, ACTIONS.index("CALL_OUTREACH")] = 0.18
    probs[:, ACTIONS.index("LINKEDIN_TOUCH")] = 0.10
    probs[:, ACTIONS.index("DEMO_OFFER")] = 0.18
    probs[:, ACTIONS.index("EXEC_SPONSOR_OUTREACH")] = 0.07
    probs[:, ACTIONS.index("PRICING_CONCESSION")] = 0.07
    probs[:, ACTIONS.index("TECHNICAL_WORKSHOP")] = 0.10

    # Enterprise: more exec + workshop
    ent_mask = df["segment"].eq("ENT").to_numpy()
    probs[ent_mask, ACTIONS.index("EXEC_SPONSOR_OUTREACH")] += 0.08
    probs[ent_mask, ACTIONS.index("TECHNICAL_WORKSHOP")] += 0.06
    probs[ent_mask, ACTIONS.index("EMAIL_SEQUENCE")] -= 0.05

    # High trial activity: demo + workshop
    high_trial = df["trial_events_30"].to_numpy() >= np.quantile(
        df["trial_events_30"].to_numpy(), 0.80
    )
    probs[high_trial, ACTIONS.index("DEMO_OFFER")] += 0.06
    probs[high_trial, ACTIONS.index("TECHNICAL_WORKSHOP")] += 0.04

    # High responses: call outreach works well
    high_resp = df["responses_30"].to_numpy() >= 2
    probs[high_resp, ACTIONS.index("CALL_OUTREACH")] += 0.06

    # Late-ish buying signals: pricing shows up
    high_intent = intent_score >= np.quantile(intent_score, 0.80)
    probs[high_intent, ACTIONS.index("PRICING_CONCESSION")] += 0.04

    # Normalize
    probs = np.clip(probs, 1e-6, None)
    probs = probs / probs.sum(axis=1, keepdims=True)

    action_idx = np.array([rng.choice(len(ACTIONS), p=probs[i]) for i in range(len(df))])
    action = np.array(ACTIONS, dtype=object)[action_idx]

    # Actions occur in the ~2 weeks leading up to as_of_date
    action_date = pd.to_datetime(cfg.as_of_date) - pd.to_timedelta(
        rng.integers(0, 14, len(df)), unit="D"
    )

    out = pd.DataFrame(
        {
            "account_id": df["account_id"].to_numpy(),
            "as_of_date": cfg.as_of_date,
            "treated": treated.astype(int),
            "propensity_any": propensity_any.astype(float),
            "action_type": np.where(treated, action, None),
            "action_date": np.where(treated, action_date.date, None),
            "intent_score": intent_score.astype(float),
        }
    )
    return out


def _simulate_outcomes(
    rng: np.random.Generator,
    cfg: SalesSynthConfig,
    accounts: pd.DataFrame,
    feats: pd.DataFrame,
    actions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Outcome: close-won within horizon.
    We also simulate an opportunity amount (ACV) and compute expected revenue.

    Important: action assignment is biased (propensity), and treatment effects are heterogeneous.
    """
    df = accounts.merge(feats, on="account_id", how="left").merge(
        actions, on=["account_id", "as_of_date"], how="left"
    )

    seg = df["segment"].to_numpy()
    affinity = df["intent_affinity"].to_numpy()
    acv = df["acv_potential"].to_numpy().astype(float)

    # Base conversion probability (counterfactual with no action)
    # Driven by intent, trial activity, responses, segment.
    z0 = (
        -2.25
        + 2.0 * df["intent_score"].to_numpy()
        + 0.015 * np.log1p(df["trial_events_30"].to_numpy())
        + 0.12 * np.log1p(df["responses_30"].to_numpy() + 1.0)
        + 0.06 * (seg == "MM").astype(float)
        + 0.12 * (seg == "ENT").astype(float)
        + 0.10 * affinity
    )
    p0 = sigmoid(z0)

    treated = df["treated"].to_numpy().astype(bool)
    action = df["action_type"].fillna("").to_numpy()

    # Base uplift by action (probability points)
    base_uplift = {
        "EMAIL_SEQUENCE": 0.010,
        "CALL_OUTREACH": 0.018,
        "LINKEDIN_TOUCH": 0.012,
        "DEMO_OFFER": 0.030,
        "EXEC_SPONSOR_OUTREACH": 0.028,
        "PRICING_CONCESSION": 0.035,
        "TECHNICAL_WORKSHOP": 0.032,
        "": 0.000,
    }
    te = np.vectorize(base_uplift.get)(action).astype(float)

    # Heterogeneity:
    intent = df["intent_score"].to_numpy()
    trial = df["trial_events_30"].to_numpy()
    responses = df["responses_30"].to_numpy()

    # Demo/workshop work best with high intent + strong trial activity
    te += np.where(
        (action == "DEMO_OFFER") & (intent >= 0.65) & (trial >= np.quantile(trial, 0.65)),
        0.018,
        0.0,
    )
    te += np.where(
        (action == "TECHNICAL_WORKSHOP") & (trial >= np.quantile(trial, 0.75)), 0.020, 0.0
    )

    # Exec sponsor works mainly for enterprise
    te += np.where((action == "EXEC_SPONSOR_OUTREACH") & (seg == "ENT"), 0.020, 0.0)
    te -= np.where((action == "EXEC_SPONSOR_OUTREACH") & (seg == "SMB"), 0.008, 0.0)

    # Calls work better when there's been some responsiveness
    te += np.where((action == "CALL_OUTREACH") & (responses >= 2), 0.010, 0.0)

    # Pricing concession increases win prob late,
    # but we’ll treat it as reducing realized revenue later
    te += np.where((action == "PRICING_CONCESSION") & (intent >= 0.70), 0.012, 0.0)
    te -= np.where((action == "PRICING_CONCESSION") & (intent < 0.45), 0.010, 0.0)

    # Email can annoy very low intent (tiny negative)
    te -= np.where((action == "EMAIL_SEQUENCE") & (intent < 0.25), 0.006, 0.0)

    # Apply only if treated
    te = te * treated.astype(float)

    p1 = np.clip(p0 + te, 0.001, 0.995)
    won = (rng.uniform(size=len(df)) < p1).astype(int)

    # Realized revenue: if pricing concession used, apply discount factor
    discount_factor = np.where(
        action == "PRICING_CONCESSION", rng.uniform(0.78, 0.92, size=len(df)), 1.0
    )
    realized_revenue = won.astype(float) * acv * discount_factor

    # Expected revenue (use observed p)
    expected_revenue = p1 * acv * discount_factor

    close_date = pd.to_datetime(cfg.as_of_date) + pd.to_timedelta(cfg.horizon_days, unit="D")

    return pd.DataFrame(
        {
            "account_id": df["account_id"].to_numpy(),
            "as_of_date": cfg.as_of_date,
            "close_date": close_date.date(),
            "won": won.astype(int),
            "acv": acv.round(2),
            "p_win_no_action": p0.astype(float),
            "p_win_observed": p1.astype(float),
            "realized_revenue": realized_revenue.round(2),
            "expected_revenue_observed": expected_revenue.round(2),
        }
    )


def generate(
    cfg: SalesSynthConfig,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    rng = np.random.default_rng(cfg.seed)
    accounts = _make_accounts(rng, cfg)
    signals = _make_signals_daily(rng, cfg, accounts)
    touches = _make_touches(rng, cfg, accounts, signals)
    feats = _make_features_as_of(cfg, signals, touches)
    actions = _assign_actions(rng, cfg, accounts, feats)
    outcomes = _simulate_outcomes(rng, cfg, accounts, feats, actions)
    return accounts, signals, touches, actions, outcomes


def load_to_duckdb(
    accounts: pd.DataFrame,
    signals: pd.DataFrame,
    touches: pd.DataFrame,
    actions: pd.DataFrame,
    outcomes: pd.DataFrame,
    db_path=None,
) -> None:
    with connect(db_path=db_path) as con:
        con.register("df_accounts", accounts)
        con.register("df_signals", signals)
        con.register("df_touches", touches)
        con.register("df_actions", actions)
        con.register("df_outcomes", outcomes)

        con.execute("CREATE OR REPLACE TABLE raw_accounts AS SELECT * FROM df_accounts")
        con.execute("CREATE OR REPLACE TABLE raw_signals_daily AS SELECT * FROM df_signals")
        con.execute("CREATE OR REPLACE TABLE raw_touches AS SELECT * FROM df_touches")
        con.execute("CREATE OR REPLACE TABLE raw_actions AS SELECT * FROM df_actions")
        con.execute("CREATE OR REPLACE TABLE raw_outcomes AS SELECT * FROM df_outcomes")


def run(cfg: SalesSynthConfig, db_path=None) -> dict[str, int]:
    accounts, signals, touches, actions, outcomes = generate(cfg)
    load_to_duckdb(accounts, signals, touches, actions, outcomes, db_path=db_path)
    return {
        "raw_accounts": len(accounts),
        "raw_signals_daily": len(signals),
        "raw_touches": len(touches),
        "raw_actions": len(actions),
        "raw_outcomes": len(outcomes),
    }
