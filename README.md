# Sales Next Best Action (NBA) Decision Intelligence System

## Overview
This project demonstrates an end-to-end Decision Intelligence Platform for sales optimization — built from scratch using synthetic data to showcase data science, analytics engineering, machine learning, and MLOps best practices.

The goal is to identify the Next Best Action (NBA) for each sales account that maximizes incremental revenue impact while accounting for action cost and treatment bias.  
The project simulates realistic sales data, develops causal uplift models, and serves recommendations through an interactive Streamlit dashboard.

---

## Architecture
Data Generation → Warehouse (DuckDB) → Feature Engineering → Uplift Modeling → Decision Dashboard → Monitoring


**Layers:**
1. **Data Generation:** Synthetic sales dataset (accounts, actions, wins, revenue).  
2. **Warehouse:** DuckDB used for persistence and SQL-style transformations.  
3. **Modeling:** Gradient Boosting models (baseline + per-action T-Learners).  
4. **Decisioning:** Streamlit dashboard for action recommendations.  
5. **Monitoring:** Notebook to track realized vs. predicted KPIs and drift.

---

## Tech Stack

| Category | Tools |
|-----------|--------|
| Data Storage | DuckDB |
| Data Engineering | pandas, dbt-style SQL |
| Modeling | scikit-learn, GradientBoostingClassifier |
| Serving | Streamlit |
| Monitoring | scipy, matplotlib, seaborn |
| Language | Python 3.9 |

---

## Data Pipeline

1. **`generate_synthetic_sales.py`**
   - Creates accounts with segments (SMB/MM/ENT), industries, and ACV potential.
   - Simulates product usage, touches, actions, and final outcomes (`won`, `realized_revenue`).

2. **`build_training_set.py`**
   - Joins raw tables (`raw_accounts`, `raw_actions`, `raw_outcomes`) into a clean modeling table `train_sales_nba`.
   - Performs feature aggregation and quality checks.

3. **Warehouse Output:**
   - `data/warehouse.duckdb` with schemas:  
     `raw_*`, `train_sales_nba`, `nba_recommendations`.

---

## Modeling Approach

### **Baseline Win Propensity**
- Predicts win likelihood *without* sales actions.  
- AUC ≈ **0.48**, confirming weak signal — actions drive outcomes.

### **Uplift Modeling**
- Per-action **T-Learners** compare treated vs. control outcomes.  
- Predicts incremental lift:  
  `uplift = P(win | treated) – P(win | control)`  
- Converts uplift into expected value (EV):  
  `EV = uplift × ACV – cost(action)`.

### **Results**
- **8 action types modeled**
- **5,000 accounts** with unique recommendations
- **Total expected incremental revenue:** **$208.5M**
- **Average EV per account:** **$41.7K**

---

## Key Insights from EDA

| Segment | Treated Lift | Interpretation |
|----------|--------------|----------------|
| SMB | +5.3 pp | Sales actions most impactful — lift nearly matches ENT baseline. |
| MM | +0.8 pp | Marginal lift — moderate accounts somewhat responsive. |
| ENT | –0.4 pp | Large accounts already convert; less incremental benefit. |

| ACV Band | Lift |
|-----------|------|
| Low | +4.6 pp |
| Med-Low | +0.9 pp |
| Med-High | **+6.4 pp** |
| High | +2.2 pp |

**Top performing actions:**  
`DEMO_OFFER`, `CALL_OUTREACH`, and `TECHNICAL_WORKSHOP` show the highest net uplift.

---

## Dashboard

**File:** `nba/serving/dashboard_app.py`

### Tabs
1. **Recommendations**
   - Interactive filters (Action, EV range)
   - KPIs: total EV, average EV, total recommendations
   - Segment × Action heatmap
   - Top accounts and CSV export

2. **Model Health**
   - Displays performance and drift metrics from monitoring notebook.
   - KPIs: predicted EV, actual revenue, realization ratio.

**Launch locally:**
```bash
streamlit run nba/serving/dashboard_app.py
```
Then visit http://localhost:8501

---

## Monitoring & Evaluation

Notebook: `notebooks/04_model_monitoring.ipynb`

- Compares predicted vs. realized revenue.
- Saves key metrics to:
    - model_performance_summary.csv
    - feature_drift_report.csv
- Includes KS test and drift visualization.
- Gracefully handles static synthetic data.

Example output:
```
Predicted Incremental Revenue: $208,538,803
Actual Realized Revenue: $208,538,803
Realization Ratio: 1.00
```
---

## Repository Structure
```markdown
nba-sales/
│
├── data/
│ ├── warehouse.duckdb
│ └── artifacts/
│ ├── baseline_win_model.pkl
│ ├── uplift_predictions.parquet
│ ├── nba_recommendations.parquet
│ └── model_performance_summary.csv
│
├── nba/
│ ├── config.py
│ ├── pipelines/
│ │ └── generate_synthetic_sales.py
│ ├── features/
│ │ └── build_training_set.py
│ ├── modeling/
│ │ ├── train_baseline.py
│ │ └── train_uplift.py
│ ├── decisioning/
│ │ └── recommend_actions.py
│ └── serving/
│ └── dashboard_app.py
│
├── notebooks/
│ ├── 01_exploration_sales_nba.ipynb
│ ├── 02_model_baseline_win_prob.ipynb
│ ├── 03_model_uplift_per_action.ipynb
│ └── 04_model_monitoring.ipynb
│
└── README.txt
```

---
## Setup and Reproduction

```bash
# Create environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Generate synthetic data
python scripts/generate_synth.py

# Build training dataset
python scripts/build_training_set.py

# Run dashboard
streamlit run nba/serving/dashboard_app.py
```

---

## Next Steps
| Area               | Enhancement                                                     |
| ------------------ | --------------------------------------------------------------- |
| **Optimization**   | Add a knapsack model to allocate limited sales capacity.        |
| **Explainability** | Integrate SHAP for per-feature uplift attribution.              |
| **Temporal Drift** | Simulate quarterly data to visualize evolving performance.      |
| **Deployment**     | Deploy Streamlit app to Streamlit Cloud or Hugging Face Spaces. |

---


## References
- Lo, V., “The True Lift Model — A Novel Data Mining Approach to Response Modeling in Database Marketing,” SIGKDD, 2002.
- Radcliffe, N., “Using Uplift Models to Optimize Direct Marketing.”
- DuckDB Documentation: https://duckdb.org/
--

---

<p align="center">
  <b>Eric McKnight</b> • Data Scientist<br>
  <i>Personal Data Science Portfolio Project</i><br>
  <a href="https://github.com/emcknight">GitHub</a> • 
  <a href="https://www.linkedin.com/in/eric-mcknight1/">LinkedIn</a>
</p>


---
