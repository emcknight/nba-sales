from nba.pipelines.generate_synthetic import SalesSynthConfig, run


def main() -> None:
    cfg = SalesSynthConfig()
    counts = run(cfg)
    print("✅ Sales synthetic data generated and loaded into DuckDB:")
    for k, v in counts.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
