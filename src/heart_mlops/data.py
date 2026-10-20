
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]


def load_raw_data() -> pd.DataFrame:
    """Download and load the processed Cleveland Heart Disease dataset."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_file = RAW_DATA_DIR / "processed.cleveland.data"

    if not raw_file.exists():
        df = pd.read_csv(
            UCI_URL,
            header=None,
            names=COLUMNS,
            na_values="?",
        )
        df.to_csv(raw_file, index=False)
    else:
        df = pd.read_csv(raw_file, na_values="?")

    return df


def prepare_data() -> pd.DataFrame:
    """Clean the dataset and convert the target into binary labels."""
    df = load_raw_data()

    if not set(COLUMNS).issubset(df.columns):
        raise ValueError("Dataset does not contain the expected columns.")

    df = df.dropna().copy()

    # Original target: 0 = no disease; 1–4 = disease.
    df["target"] = (df["target"] > 0).astype(int)

    if df.empty:
        raise ValueError("No rows remain after removing missing values.")

    if not set(df["target"].unique()).issubset({0, 1}):
        raise ValueError("Target must contain only binary labels: 0 and 1.")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_file = PROCESSED_DATA_DIR / "heart_disease.csv"
    df.to_csv(output_file, index=False)

    print(f"Processed dataset saved to: {output_file}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("Target distribution:")
    print(df["target"].value_counts().sort_index())

    return df


if __name__ == "__main__":
    prepare_data()
