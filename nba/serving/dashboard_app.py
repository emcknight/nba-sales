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
        # Try loading from Parquet
        df = pd.read_parquet(settings.artifacts_dir / "nba_recommendations.parquet")
    except Exception:
        # Fallback to DuckDB
        with duckdb.connect(str(settings.db_path)) as con:
            df = con.execute("SELECT * FROM nba_recommendations").fetchdf()
    return df


df = load_data()

def derive_segment(acv):
    if acv < 50_000:
        return "SMB"
    elif acv < 250_000:
        return "MM"
    else:
        return "ENT"

df["segment"] = df["acv"].apply(derive_segment)

# ----------------------------------------------------
# Tabs
# ----------------------------------------------------
tab1, tab2 = st.tabs(["Recommendations", "Model Health"])

# ====================================================
# Tab 1: Recommendations Dashboard
# ====================================================
with tab1:

    # Sidebar Filters
    st.sidebar.header("Filters")

    actions = st.sidebar.multiselect(
        "Select Action Types:",
        sorted(df["action_type"].unique().tolist()),
        default=sorted(df["action_type"].unique().tolist())
    )

    min_ev, max_ev = st.sidebar.slider(
        "Expected Value Range ($)",
        float(df["expected_value"].min()),
        float(df["expected_value"].max()),
        (float(df["expected_value"].quantile(0.05)),
         float(df["expected_value"].quantile(0.95)))
    )

    filtered_df = df[
        df["action_type"].isin(actions)
        & (df["expected_value"].between(min_ev, max_ev))
    ]

    # Summary Metrics
    st.title("Next Best Action — Decision Intelligence Dashboard")

    total_ev = filtered_df["expected_value"].sum()
    avg_ev = filtered_df["expected_value"].mean()
    num_recs = len(filtered_df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Expected Incremental Revenue", f"${total_ev:,.0f}")
    col2.metric("Average Expected Value", f"${avg_ev:,.0f}")
    col3.metric("Total Recommendations", f"{num_recs:,}")

    # Visualizations
    st.subheader("Expected Value by Action Type")
    ev_by_action = (
        filtered_df.groupby("action_type")["expected_value"]
        .sum()
        .sort_values(ascending=False)
    )
    st.bar_chart(ev_by_action)

    # Segment-Level Summaries
    st.subheader("Segment-Level Summaries")

    if "segment" in df.columns:
        seg_summary = (
            filtered_df.groupby(["segment", "action_type"])
            .agg(
                total_ev=("expected_value", "sum"),
                avg_ev=("expected_value", "mean"),
                num_accounts=("account_id", "nunique"),
            )
            .reset_index()
            .sort_values("total_ev", ascending=False)
        )

        st.markdown("**Total Expected Incremental Revenue by Segment and Action Type**")
        st.dataframe(seg_summary)

        pivot = (
            seg_summary.pivot(
                index="segment", columns="action_type", values="total_ev"
            )
            .fillna(0)
        )
        st.markdown("**Revenue Heatmap (Segment × Action Type)**")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".0f", ax=ax)
        plt.title("Total Expected Incremental Revenue by Segment and Action Type")
        plt.xlabel("Action Type")
        plt.ylabel("Segment")
        st.pyplot(fig)
    else:
        st.warning("No 'segment' column found in data — segment-level summaries unavailable.")

    st.subheader("Distribution of Expected Value")
    sns.histplot(filtered_df["expected_value"], bins=40, kde=True)
    st.pyplot(plt.gcf())

    # Top Accounts
    st.subheader("Top 20 Accounts by Expected Value")
    top_accounts = filtered_df.sort_values("expected_value", ascending=False).head(20)
    st.dataframe(
        top_accounts[["account_id", "action_type", "expected_value", "uplift", "acv"]]
    )

    # Export
    st.download_button(
        "Download Filtered Recommendations (CSV)",
        data=filtered_df.to_csv(index=False),
        file_name="nba_recommendations_filtered.csv",
        mime="text/csv",
    )

# ====================================================
# Tab 2: Model Health & Monitoring
# ====================================================
with tab2:
    st.title("Model Health & Monitoring")

    perf_path = settings.artifacts_dir / "model_performance_summary.csv"
    drift_path = settings.artifacts_dir / "feature_drift_report.csv"

    try:
        perf_df = pd.read_csv(perf_path)
        st.subheader("Performance Summary")
        st.dataframe(perf_df)

        # KPI boxes
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
