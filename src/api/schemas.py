# src/api/schemas.py
"""
Pydantic models for API request and response validation.

Why Pydantic?
FastAPI uses Pydantic to automatically:
- Validate incoming request data (wrong types -> 422 error)
- Generate OpenAPI docs at /docs
- Serialize response data to JSON
"""

from pydantic import BaseModel, Field


class BikeFeatures(BaseModel):
    """
    Input features required to make a prediction.
    These match exactly the columns remaining after preprocessing.
    """
    season: int = Field(..., ge=1, le=4, description="1=spring, 2=summer, 3=fall, 4=winter")
    yr: int = Field(..., ge=0, le=1, description="0=2011, 1=2012")
    mnth: int = Field(..., ge=1, le=12)
    hr: int = Field(..., ge=0, le=23)
    holiday: int = Field(..., ge=0, le=1)
    weekday: int = Field(..., ge=0, le=6)
    workingday: int = Field(..., ge=0, le=1)
    weathersit: int = Field(..., ge=1, le=4, description="1=clear, 2=mist, 3=light snow/rain, 4=heavy rain")
    temp: float = Field(..., ge=0.0, le=1.0, description="Normalized temperature")
    atemp: float = Field(..., ge=0.0, le=1.0, description="Normalized feeling temperature")
    hum: float = Field(..., ge=0.0, le=1.0, description="Normalized humidity")
    windspeed: float = Field(..., ge=0.0, le=1.0, description="Normalized wind speed")


class PredictionResponse(BaseModel):
    """Output returned by the /predict endpoint."""
    model_config = {"protected_namespaces": ()}
    predicted_cnt: float = Field(..., description="Predicted number of bike rentals")
    model_version: str = Field(..., description="MLflow model version used for prediction")
    latency_ms: float = Field(..., description="Prediction latency in milliseconds")


class HealthResponse(BaseModel):
    """Output returned by the /health endpoint."""
    model_config = {"protected_namespaces": ()}
    status: str
    model_loaded: bool
    model_version: str