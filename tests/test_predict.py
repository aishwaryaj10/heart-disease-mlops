
import pytest

from src.heart_mlops.predict import (
    FEATURE_COLUMNS,
    predict_heart_disease,
)


@pytest.fixture
def valid_sample():
    return {
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


def test_prediction_returns_valid_output(valid_sample):
    result = predict_heart_disease(valid_sample)

    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["probability"] <= 1.0


def test_missing_feature_is_rejected(valid_sample):
    valid_sample.pop("age")

    with pytest.raises(ValueError, match="Missing features"):
        predict_heart_disease(valid_sample)


def test_extra_feature_is_rejected(valid_sample):
    valid_sample["unexpected_feature"] = 123

    with pytest.raises(ValueError, match="Unexpected features"):
        predict_heart_disease(valid_sample)


def test_all_expected_features_are_defined():
    assert len(FEATURE_COLUMNS) == 13
    assert len(set(FEATURE_COLUMNS)) == 13
