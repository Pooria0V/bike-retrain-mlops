# src/monitoring/prediction_logger.py
"""
Logs every prediction request to a CSV file for later analysis.
This gives us an audit trail of what the model predicted over time,
which is useful for detecting model degradation in production.
"""

import csv
import uuid
from datetime import datetime, timezone
from pathlib import Path

LOGS_DIR = Path("logs")
LOG_FILE = LOGS_DIR / "predictions.csv"

# CSV columns
FIELDNAMES = [
    "prediction_id",
    "timestamp",
    "season", "yr", "mnth", "hr",
    "holiday", "weekday", "workingday", "weathersit",
    "temp", "atemp", "hum", "windspeed",
    "predicted_cnt",
    "latency_ms",
]


def init_log_file():
    """
    Create the log file with headers if it doesn't exist yet.
    Called once when the API starts up.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if not LOG_FILE.exists():
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
        print(f"Prediction log file created: {LOG_FILE}")


def log_prediction(input_data: dict, predicted_cnt: float, latency_ms: float):
    """
    Append one prediction record to the CSV log file.

    Why CSV and not a database?
    For a learning project, CSV is simple and transparent —
    you can open it in Excel or pandas instantly.
    In production, you'd use a time-series DB like InfluxDB or TimescaleDB.

    Args:
        input_data   : the feature dict sent to the API
        predicted_cnt: model output (predicted bike count)
        latency_ms   : how long the prediction took in milliseconds
    """
    record = {
        "prediction_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "predicted_cnt": round(predicted_cnt, 2),
        "latency_ms": round(latency_ms, 3),
        **input_data,
    }

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(record)