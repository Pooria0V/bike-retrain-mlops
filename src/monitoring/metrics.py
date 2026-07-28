# src/monitoring/metrics.py
"""
Prometheus metrics for operational monitoring of the FastAPI service.

How Prometheus works:
- This module defines counters and histograms
- FastAPI exposes them at GET /metrics
- Prometheus server scrapes /metrics every N seconds
- Grafana reads from Prometheus and shows dashboards

Think of it like a speedometer in a car:
- The car (FastAPI) continuously updates the speed value
- The dashboard (Grafana) reads and displays it
"""

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Counter: only goes up — total number of predictions served
PREDICTION_COUNTER = Counter(
    name="bike_predictions_total",
    documentation="Total number of prediction requests served",
    labelnames=["status"],   # label: 'success' or 'error'
)

# Histogram: tracks distribution of values over time
# Here we track how long each prediction takes (in seconds)
# Buckets define the boundaries for latency groups
PREDICTION_LATENCY = Histogram(
    name="bike_prediction_latency_seconds",
    documentation="Prediction request latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

# Counter: tracks how many times the model has been retrained
RETRAIN_COUNTER = Counter(
    name="bike_retrains_total",
    documentation="Total number of model retraining events",
    labelnames=["result"],   # label: 'promoted' or 'rejected'
)


def get_metrics():
    """
    Returns current metrics in Prometheus text format.
    This is what GET /metrics endpoint exposes.
    """
    return generate_latest(), CONTENT_TYPE_LATEST