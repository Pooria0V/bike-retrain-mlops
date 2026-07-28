# src/api/main.py
"""
FastAPI service that serves the current production model.

Key design decisions:
- Model is loaded ONCE at startup (not on every request)
  because loading from MLflow registry is expensive (~seconds)
- /metrics endpoint exposes Prometheus metrics for Grafana
- /health endpoint lets load balancers check if service is alive
- Every prediction is logged to CSV for later drift analysis
"""

import os
import time
import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from contextlib import asynccontextmanager

from src.api.schemas import BikeFeatures, PredictionResponse, HealthResponse
from src.monitoring.metrics import (
    PREDICTION_COUNTER,
    PREDICTION_LATENCY,
    get_metrics,
)
from src.monitoring.prediction_logger import init_log_file, log_prediction

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
MODEL_NAME = "bike-demand-model"
CHAMPION_ALIAS = "champion"

# Global model state — loaded once at startup
model = None
model_version = "unknown"


def load_champion_model():
    """
    Load the current champion model from MLflow Model Registry.

    Why load at startup?
    Each prediction request would take 1-2 seconds if we loaded
    the model on every call. Loading once keeps latency under 50ms.
    """
    global model, model_version
    mlflow.set_tracking_uri(TRACKING_URI)

    try:
        client = mlflow.tracking.MlflowClient()
        version_info = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
        model_version = version_info.version

        model_uri = f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}"
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Model loaded: {MODEL_NAME} version {model_version}")
    except Exception as e:
        print(f"WARNING: Could not load model — {e}")
        model = None
        model_version = "unavailable"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager: runs setup before the app starts
    and teardown when it stops.

    Why lifespan instead of @app.on_event("startup")?
    FastAPI deprecated on_event in favor of lifespan.
    """
    print("Starting up — loading model and initializing log file...")
    load_champion_model()
    init_log_file()
    yield
    print("Shutting down...")


app = FastAPI(
    title="Bike Demand Prediction API",
    description="Serves the current champion model from MLflow Registry",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    """
    Health check endpoint.
    Load balancers and Docker health checks call this to verify
    the service is alive and the model is loaded.
    """
    return HealthResponse(
        status="ok" if model is not None else "degraded",
        model_loaded=model is not None,
        model_version=model_version,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(features: BikeFeatures):
    """
    Main prediction endpoint.

    Flow:
    1. Validate input (Pydantic does this automatically)
    2. Convert to DataFrame (sklearn expects this format)
    3. Run prediction and measure latency
    4. Log prediction to CSV
    5. Update Prometheus metrics
    6. Return result
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    input_dict = features.model_dump()
    input_df = pd.DataFrame([input_dict])

    start = time.perf_counter()
    prediction = model.predict(input_df)[0]
    latency_ms = (time.perf_counter() - start) * 1000

    # Log to CSV for future drift analysis
    log_prediction(
        input_data=input_dict,
        predicted_cnt=float(prediction),
        latency_ms=latency_ms,
    )

    # Update Prometheus counters
    PREDICTION_COUNTER.labels(status="success").inc()
    PREDICTION_LATENCY.observe(latency_ms / 1000)

    return PredictionResponse(
        predicted_cnt=round(float(prediction), 2),
        model_version=model_version,
        latency_ms=round(latency_ms, 3),
    )


@app.get("/metrics")
def metrics():
    """
    Prometheus metrics endpoint.
    Prometheus server scrapes this every N seconds
    and stores the time-series data for Grafana dashboards.
    """
    data, content_type = get_metrics()
    return Response(content=data, media_type=content_type)