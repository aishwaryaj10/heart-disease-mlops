
import logging
import time
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, ConfigDict, Field

from src.heart_mlops.predict import predict_heart_disease


# Log to the container's standard output so Docker can collect logs.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("heart_disease_api")

app = FastAPI(
    title="Heart Disease Prediction API",
    description="An educational ML model serving demonstration.",
    version="1.1.0",
)

# Prometheus metrics
HTTP_REQUESTS = Counter(
    "http_requests_total",
    "Total HTTP requests handled by the API",
    ["method", "endpoint", "status"],
)

HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request processing duration in seconds",
    ["method", "endpoint"],
)

PREDICTIONS = Counter(
    "heart_disease_predictions_total",
    "Total successful model predictions",
    ["prediction"],
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


@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception:
        logger.exception(
            "Unhandled request error method=%s path=%s",
            request.method,
            request.url.path,
        )
        raise
    finally:
        duration = time.perf_counter() - start_time
        path = request.url.path

        # Avoid putting arbitrary URL values into metric labels.
        if path not in {"/", "/health", "/predict", "/metrics"}:
            path = "other"

        HTTP_REQUESTS.labels(
            method=request.method,
            endpoint=path,
            status=str(status_code),
        ).inc()

        HTTP_REQUEST_DURATION.labels(
            method=request.method,
            endpoint=path,
        ).observe(duration)

        logger.info(
            "HTTP request method=%s path=%s status=%s duration_seconds=%.4f",
            request.method,
            path,
            status_code,
            duration,
        )


@app.get("/")
def home():
    return {
        "message": "Heart Disease Prediction API is running",
        "docs": "/docs",
        "metrics": "/metrics",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(data: HeartDiseaseInput):
    try:
        result = predict_heart_disease(data.model_dump())
        PREDICTIONS.labels(prediction=str(result["prediction"])).inc()
        return result
    except (ValueError, FileNotFoundError) as exc:
        logger.warning("Prediction failed error_type=%s", type(exc).__name__)
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
