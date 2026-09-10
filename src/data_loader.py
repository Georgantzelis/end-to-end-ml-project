from pathlib import Path

import pandas as pd


DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


def load_data() -> pd.DataFrame:
    """Load the raw Telco Customer Churn dataset."""
    return pd.read_csv(DATA_PATH)


if __name__ == "__main__":
    df = load_data()

    print(f"Dataset shape: {df.shape}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nChurn distribution:")
    print(df["Churn"].value_counts())

    print("\nMissing values:")
    print(df.isna().sum())