# src/tests/test_api.py
"""
Integration tests for the FastAPI service.

Why httpx and TestClient?
FastAPI's TestClient runs the app in-process without needing
a real running server — fast, isolated, and no port conflicts.

Note: these tests mock the model so they don't need
a live MLflow server or a trained model to run in CI.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import numpy as np


VALID_PAYLOAD = {
    "season": 2,
    "yr": 0,
    "mnth": 6,
    "hr": 17,
    "holiday": 0,
    "weekday": 2,
    "workingday": 1,
    "weathersit": 1,
    "temp": 0.68,
    "atemp": 0.6364,
    "hum": 0.79,
    "windspeed": 0.1343,
}


@pytest.fixture
def client():
    """
    Create a TestClient with a mocked model.

    Why mock?
    We don't want tests to depend on a running MLflow server.
    The mock replaces the real model with a fake one that always
    returns 200.0 — so we can test API logic independently.
    """
    with patch("src.api.main.model") as mock_model, \
         patch("src.api.main.model_version", "test-v1"):
        mock_model.predict.return_value = np.array([200.0])
        from src.api.main import app
        yield TestClient(app)


def test_health_endpoint(client):
    """GET /health must return 200 with model_loaded=True."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_valid_input(client):
    """POST /predict with valid input must return predicted_cnt."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_cnt" in data
    assert "model_version" in data
    assert "latency_ms" in data
    assert data["predicted_cnt"] == pytest.approx(200.0)


def test_predict_missing_field(client):
    """POST /predict with a missing field must return 422."""
    incomplete = VALID_PAYLOAD.copy()
    incomplete.pop("temp")
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_predict_invalid_season(client):
    """season must be between 1 and 4 — out-of-range must return 422."""
    bad_payload = VALID_PAYLOAD.copy()
    bad_payload["season"] = 99
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_invalid_hour(client):
    """hr must be between 0 and 23 — out-of-range must return 422."""
    bad_payload = VALID_PAYLOAD.copy()
    bad_payload["hr"] = 25
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_metrics_endpoint(client):
    """GET /metrics must return Prometheus-formatted text."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"bike_predictions_total" in response.content