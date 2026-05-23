"""FastAPI serving layer for the trained heart disease model."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline

from heart_disease_mlops.config import FEATURE_COLUMNS, MODEL_PATH
from heart_disease_mlops.features import validate_feature_frame
from heart_disease_mlops.modeling import predict_probability

LOGGER = logging.getLogger("heart_disease_mlops.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REQUEST_COUNT = Counter(
    "heart_disease_api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "heart_disease_api_request_latency_seconds",
    "API request latency in seconds",
    ["endpoint"],
)


class PatientFeatures(BaseModel):
    """Input schema for one patient record."""

    age: float = Field(..., examples=[63])
    sex: float = Field(..., examples=[1])
    cp: float = Field(..., examples=[3])
    trestbps: float = Field(..., examples=[145])
    chol: float = Field(..., examples=[233])
    fbs: float = Field(..., examples=[1])
    restecg: float = Field(..., examples=[0])
    thalach: float = Field(..., examples=[150])
    exang: float = Field(..., examples=[0])
    oldpeak: float = Field(..., examples=[2.3])
    slope: float = Field(..., examples=[0])
    ca: float | None = Field(None, examples=[0])
    thal: float | None = Field(None, examples=[1])

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the request payload to a single-row dataframe."""
        record = self.model_dump()
        return pd.DataFrame([record], columns=FEATURE_COLUMNS)


class PredictionResponse(BaseModel):
    """Output schema returned by the prediction endpoint."""

    prediction: int
    risk_label: str
    heart_disease_probability: float
    confidence: float


def load_model(model_path: Path = MODEL_PATH) -> Pipeline:
    """Load the persisted sklearn pipeline."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Train the model before starting the API."
        )
    return joblib.load(model_path)


def create_app(model_path: Path = MODEL_PATH) -> FastAPI:
    """Create the FastAPI app with a loaded model dependency."""
    @asynccontextmanager
    async def lifespan(app_instance: FastAPI) -> AsyncIterator[None]:
        app_instance.state.model = load_model(model_path)
        yield

    app = FastAPI(
        title="Heart Disease Risk Prediction API",
        version="0.1.0",
        description="Cloud-ready API for the BITS MLOps heart disease assignment.",
        lifespan=lifespan,
    )
    app.state.model_path = model_path
    app.state.model = None

    @app.middleware("http")
    async def metrics_and_logging_middleware(request: Request, call_next: Any) -> Any:
        start_time = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start_time

        endpoint = request.url.path
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed)
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=str(response.status_code),
        ).inc()
        LOGGER.info(
            "request method=%s path=%s status=%s latency=%.4fs",
            request.method,
            endpoint,
            response.status_code,
            elapsed,
        )
        return response

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/predict", response_model=PredictionResponse)
    def predict(payload: PatientFeatures) -> PredictionResponse:
        feature_df = payload.to_dataframe()
        try:
            validate_feature_frame(feature_df)
            model = app.state.model or load_model(app.state.model_path)
            probability = float(predict_probability(model, feature_df)[0])
        except Exception as exc:
            LOGGER.exception("prediction_failed")
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        prediction = int(probability >= 0.5)
        confidence = probability if prediction == 1 else 1.0 - probability
        risk_label = "high_risk" if prediction == 1 else "low_risk"

        return PredictionResponse(
            prediction=prediction,
            risk_label=risk_label,
            heart_disease_probability=round(probability, 6),
            confidence=round(confidence, 6),
        )

    return app


app = create_app()
