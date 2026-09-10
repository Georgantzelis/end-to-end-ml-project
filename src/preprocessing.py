import pandas as pd


TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw Telco Customer Churn dataset."""
    data = df.copy()

    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"].str.strip(),
        errors="coerce",
    )

    data = data.dropna(subset=["TotalCharges"]).copy()

    return data


def prepare_features_and_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare model features X and binary target y."""
    data = clean_data(df)

    X = data.drop(columns=[TARGET_COLUMN, ID_COLUMN])

    y = data[TARGET_COLUMN].map(
        {
            "No": 0,
            "Yes": 1,
        }
    )

    return X, y