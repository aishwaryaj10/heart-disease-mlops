
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.heart_mlops.predict import predict_heart_disease


app = FastAPI(
    title="Heart Disease Prediction API",
    description="An educational ML model serving demonstration.",
    version="1.0.0",
)


class HeartDiseaseInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: int = Field(ge=1, le=120)
    sex: Literal[0, 1]
    cp: int = Field(ge=1, le=4)
    trestbps: int = Field(gt=0, le=300)
    chol: int = Field(gt=0, le=1000)
    fbs: Literal[0, 1]
    restecg: int = Field(ge=0, le=2)
    thalach: int = Field(gt=0, le=300)
    exang: Literal[0, 1]
    oldpeak: float = Field(ge=0, le=20)
    slope: int = Field(ge=1, le=3)
    ca: int = Field(ge=0, le=3)
    thal: int = Field(ge=3, le=7)


class PredictionResponse(BaseModel):
    prediction: Literal[0, 1]
    probability: float


@app.get("/")
def home():
    return {
        "message": "Heart Disease Prediction API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: HeartDiseaseInput):
    try:
        return predict_heart_disease(data.model_dump())
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
