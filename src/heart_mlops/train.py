
from pathlib import Path
import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from src.heart_mlops.data import prepare_data
from src.heart_mlops.preprocessing import build_preprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MLFLOW_DIR = PROJECT_ROOT / "mlruns"

RANDOM_STATE = 42
TEST_SIZE = 0.20


def train_and_evaluate():
    """Train candidate models, evaluate them, and track results in MLflow."""
    df = prepare_data()

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
    mlflow.set_experiment("heart-disease-classification")
    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
    }

    results = []

    for model_name, estimator in candidates.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", estimator),
            ]
        )

        with mlflow.start_run(run_name=model_name):
            pipeline.fit(X_train, y_train)

            predictions = pipeline.predict(X_test)
            probabilities = pipeline.predict_proba(X_test)[:, 1]

            cv_scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=5,
                scoring="f1",
            )

            metrics = {
                "test_accuracy": accuracy_score(y_test, predictions),
                "test_precision": precision_score(
                    y_test, predictions, zero_division=0
                ),
                "test_recall": recall_score(
                    y_test, predictions, zero_division=0
                ),
                "test_f1": f1_score(
                    y_test, predictions, zero_division=0
                ),
                "test_roc_auc": roc_auc_score(y_test, probabilities),
                "cv_f1_mean": cv_scores.mean(),
            }

            mlflow.log_param("model_type", model_name)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("test_size", TEST_SIZE)
            mlflow.log_metrics(metrics)

            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                input_example=X_train.head(3),
                skops_trusted_types=[
                    "numpy.dtype",
                    "sklearn.tree._tree.Tree",
                ],
            )

            results.append(
                {
                    "model_name": model_name,
                    **metrics,
                }
            )

            print(f"\n{model_name}")
            for metric_name, metric_value in metrics.items():
                print(f"{metric_name}: {metric_value:.4f}")

    results_df = pd.DataFrame(results).sort_values(
        "cv_f1_mean",
        ascending=False,
    )
    results_df.to_csv(ARTIFACT_DIR / "model_comparison.csv", index=False)

    best_name = results_df.iloc[0]["model_name"]
    best_estimator = candidates[best_name]

    best_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", best_estimator),
        ]
    )
    best_pipeline.fit(X_train, y_train)

    joblib.dump(
        best_pipeline,
        ARTIFACT_DIR / "heart_disease_model.joblib",
    )

    metadata = {
        "selected_model": best_name,
        "target_column": "target",
        "positive_class": 1,
        "feature_columns": list(X.columns),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
    }
    (ARTIFACT_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nModel comparison:")
    print(results_df.to_string(index=False))
    print(f"\nSelected model: {best_name}")
    print(f"Saved model to: {ARTIFACT_DIR / 'heart_disease_model.joblib'}")
    print(f"MLflow tracking directory: {MLFLOW_DIR}")


if __name__ == "__main__":
    train_and_evaluate()
