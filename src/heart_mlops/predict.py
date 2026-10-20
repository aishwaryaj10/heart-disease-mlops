
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "heart_disease_model.joblib"

FEATURE_COLUMNS = [
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
]


def load_model():
    """Load the trained model pipeline from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run the training module first."
        )

    return joblib.load(MODEL_PATH)


def predict_heart_disease(input_data: dict) -> dict:
    """Return a binary prediction and estimated probability."""
    missing = set(FEATURE_COLUMNS) - set(input_data)
    extra = set(input_data) - set(FEATURE_COLUMNS)

    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")

    if extra:
        raise ValueError(f"Unexpected features: {sorted(extra)}")

    row = pd.DataFrame(
        [{name: input_data[name] for name in FEATURE_COLUMNS}],
        columns=FEATURE_COLUMNS,
    )

    if row.isnull().any().any():
        raise ValueError("Input features cannot be missing.")

    model = load_model()
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0][1])

    return {
        "prediction": prediction,
        "probability": round(probability, 4),
    }

if __name__ == "__main__":
    sample = {
        "age": 55,
        "sex": 1,
        "cp": 2,
        "trestbps": 130,
        "chol": 250,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 2,
        "ca": 0,
        "thal": 2,
    }

    result = predict_heart_disease(sample)
    print("Prediction result:", result)