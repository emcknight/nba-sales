from nba.features.build_training_set import build_training_set

def main() -> None:
    df = build_training_set()
    print(f"✅ Training set built: {len(df)} rows")
    print(df[["account_id", "action_type", "won", "expected_revenue_observed"]].head())

if __name__ == "__main__":
    main()
