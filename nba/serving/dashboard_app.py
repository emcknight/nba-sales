import streamlit as st
import pandas as pd
import duckdb
import matplotlib.pyplot as plt
import seaborn as sns
from nba.config import settings

# ----------------------------------------------------
# Load Data
# ----------------------------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_parquet(settings.artifacts_dir / "nba_recommendations.parquet")
    except Exception:
        with duckdb.connect(str(settings.db_path)) as con:
            df = con.execute("SELECT * FROM nba_recommendations").fetchdf()
    return df


df = load_data()

# ----------------------------------------------------
# Derive Segment
# ----------------------------------------------------
def derive_segment(acv):
    if acv < 50_000:
        return "SMB"
    elif acv < 250_000:
        return "MM"
    else:
        return "ENT"

df["segment"] = df["acv"].apply(derive_segment)

# Human-readable action names
action_labels = {
    "CALL_OUTREACH": "Call Outreach",
    "DEMO_OFFER": "Product Demo",
    "EMAIL_SEQUENCE": "Email Sequence",
    "EXEC_SPONSOR_OUTREACH": "Executive Sponsor Outreach",
    "LINKEDIN_TOUCH": "LinkedIn Touch",
    "PRICING_CONCESSION": "Pricing Concession",
    "TECHNICAL_WORKSHOP": "Technical Workshop",
    "0": "No Action"
}
df["action_display"] = df["action_type"].map(action_labels).fillna(df["action_type"])

# ----------------------------------------------------
# Tabs
# ----------------------------------------------------
tab1, tab2 = st.tabs(["Recommendations", "Model Health"])

# ====================================================
# TAB 1: Recommendations
# ====================================================
with tab1:

    st.sidebar.header("Filters")

    # 1️⃣ Multi-select dropdown (checkbox style)
    actions = st.sidebar.multiselect(
        "Select Action Types:",
        options=sorted(df["action_display"].unique().tolist()),
        default=sorted(df["action_display"].unique().tolist())
    )

    # Map back to internal action_type
    selected_actions = df[df["action_display"].isin(actions)]["action_type"].unique().tolist()

    # 2️⃣ Expected Value Range Buckets + Slider
    st.sidebar.markdown("### Expected Value Range")
    ev_bucket = st.sidebar.radio(
        "Select Value Bucket:",
        ["All", "< $10K", "$10K–$50K", "$50K–$100K", "> $100K"],
        index=0
    )

    # Compute slider range
    min_ev, max_ev = st.sidebar.slider(
        "Expected Value ($ Range)",
        float(df["expected_value"].min()),
        float(df["expected_value"].max()),
        (
            float(df["expected_value"].quantile(0.05)),
            float(df["expected_value"].quantile(0.95)),
        ),
    )

    # 3️⃣ Segment Filter
    segments = sorted(df["segment"].unique().tolist())
    selected_segments = st.sidebar.multiselect(
        "Select Segments:",
        options=segments,
        default=segments
    )

    # Filtering logic
    filtered_df = df[
        df["action_type"].isin(selected_actions)
        & df["segment"].isin(selected_segments)
        & (df["expected_value"].between(min_ev, max_ev))
    ]

    if ev_bucket != "All":
        if ev_bucket == "< $10K":
            filtered_df = filtered_df[filtered_df["expected_value"] < 10_000]
        elif ev_bucket == "$10K–$50K":
            filtered_df = filtered_df[
                (filtered_df["expected_value"] >= 10_000)
                & (filtered_df["expected_value"] < 50_000)
            ]
        elif ev_bucket == "$50K–$100K":
            filtered_df = filtered_df[
                (filtered_df["expected_value"] >= 50_000)
                & (filtered_df["expected_value"] < 100_000)
            ]
        else:
            filtered_df = filtered_df[filtered_df["expected_value"] >= 100_000]

    # Summary KPIs
    st.title("Next Best Action — Decision Intelligence Dashboard")

    total_ev = filtered_df["expected_value"].sum()
    avg_ev = filtered_df["expected_value"].mean()
    num_recs = len(filtered_df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Expected Incremental Revenue", f"${total_ev:,.2f}")
    col2.metric("Average Expected Value", f"${avg_ev:,.2f}")
    col3.metric("Total Recommendations", f"{num_recs:,}")

    # Expected Value by Action Type
    st.subheader("Expected Value by Action Type")
    ev_by_action = (
        filtered_df.groupby("action_display")["expected_value"]
        .sum()
        .sort_values(ascending=False)
    )
    st.bar_chart(ev_by_action)

    # 4️⃣ Segment-Level Summaries (sorted, formatted)
    st.subheader("Segment-Level Summaries")

    seg_summary = (
        filtered_df.groupby(["segment", "action_display"])
        .agg(
            total_ev=("expected_value", "sum"),
            avg_ev=("expected_value", "mean"),
            num_accounts=("account_id", "nunique"),
        )
        .reset_index()
        .sort_values(["segment", "action_display"], ascending=[True, True])
    )

    seg_summary["total_ev"] = seg_summary["total_ev"].apply(
        lambda x: f"${x:,.2f}"
    )
    seg_summary["avg_ev"] = seg_summary["avg_ev"].apply(
        lambda x: f"${x:,.2f}"
    )

    st.markdown("**Total Expected Incremental Revenue by Segment and Action Type**")
    st.dataframe(seg_summary)

    # 5️⃣ Heatmap (show $K)
    st.markdown("**Revenue Heatmap (Segment × Action Type)**")
    pivot = (
        filtered_df.groupby(["segment", "action_display"])["expected_value"]
        .sum()
        .reset_index()
        .pivot(index="segment", columns="action_display", values="expected_value")
        .fillna(0)
    )

    # Convert to $K for heatmap labels
    pivot_display = pivot.applymap(lambda x: x / 1000)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        pivot_display,
        cmap="YlGnBu",
        annot=True,
        fmt=".0f",
        ax=ax,
        cbar_kws={"label": "Expected Value ($K)"},
    )
    plt.title("Total Expected Incremental Revenue (in $K)")
    plt.xlabel("Action Type")
    plt.ylabel("Segment")
    st.pyplot(fig)

    # Distribution of Expected Value
    st.subheader("Distribution of Expected Value")
    sns.histplot(filtered_df["expected_value"], bins=40, kde=True)
    st.pyplot(plt.gcf())

    # Top Accounts
    st.subheader("Top 20 Accounts by Expected Value")
    top_accounts = (
        filtered_df.sort_values("expected_value", ascending=False)
        .head(20)
        .copy()
    )
    top_accounts["expected_value"] = top_accounts["expected_value"].apply(
        lambda x: f"${x:,.2f}"
    )
    st.dataframe(
        top_accounts[
            ["account_id", "segment", "action_display", "expected_value", "uplift", "acv"]
        ]
    )

    # Export CSV
    st.download_button(
        "Download Filtered Recommendations (CSV)",
        data=filtered_df.to_csv(index=False),
        file_name="nba_recommendations_filtered.csv",
        mime="text/csv",
    )

# ====================================================
# TAB 2: Model Health
# ====================================================
with tab2:
    st.title("Model Health & Monitoring")

    perf_path = settings.artifacts_dir / "model_performance_summary.csv"
    drift_path = settings.artifacts_dir / "feature_drift_report.csv"

    try:
        perf_df = pd.read_csv(perf_path)
        st.subheader("Performance Summary")
        st.dataframe(perf_df)

        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted EV", f"${perf_df['predicted_total_ev'].iloc[0]:,.0f}")
        col2.metric("Actual Revenue", f"${perf_df['actual_total_revenue'].iloc[0]:,.0f}")
        col3.metric("Realization Ratio", f"{perf_df['realization_ratio'].iloc[0]:.2f}")
    except FileNotFoundError:
        st.warning("Performance summary file not found. Run the monitoring notebook first.")

    st.markdown("---")

    try:
        drift_df = pd.read_csv(drift_path)
        st.subheader("Feature Drift Report")
        st.dataframe(drift_df)
    except FileNotFoundError:
        st.warning("Drift report file not found. Run the monitoring notebook first.")

    st.markdown("---")
    st.markdown(
        "This section displays performance and drift metrics produced by the monitoring notebook. "
        "Use it to assess whether model predictions remain stable and consistent over time."
    )
