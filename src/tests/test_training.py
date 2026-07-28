# src/tests/test_training.py
"""
Unit tests for training module.

We test the helper functions independently from MLflow
so tests run fast without needing a running MLflow server.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.training.train import prepare_X_y, compute_metrics

CHUNKS_DIR = Path("data/monthly_chunks")


def make_sample_df() -> pd.DataFrame:
    """
    Create a small synthetic DataFrame that mimics the real dataset.
    Used to test functions without loading the full 17k-row dataset.
    """
    return pd.DataFrame({
        "dteday": ["2011-01-01"] * 10,
        "season": [1] * 10,
        "yr": [0] * 10,
        "mnth": [1] * 10,
        "hr": list(range(10)),
        "holiday": [0] * 10,
        "weekday": [1] * 10,
        "workingday": [1] * 10,
        "weathersit": [1] * 10,
        "temp": [0.5] * 10,
        "atemp": [0.5] * 10,
        "hum": [0.6] * 10,
        "windspeed": [0.2] * 10,
        "cnt": list(range(50, 150, 10)),
    })


def test_prepare_X_y_separates_target():
    """X must not contain 'cnt', y must be a Series of cnt values."""
    df = make_sample_df()
    X, y = prepare_X_y(df)

    assert "cnt" not in X.columns
    assert list(y) == list(range(50, 150, 10))


def test_prepare_X_y_drops_dteday():
    """dteday must be dropped from X since it's not a usable feature."""
    df = make_sample_df()
    X, y = prepare_X_y(df)
    assert "dteday" not in X.columns


def test_compute_metrics_perfect_prediction():
    """
    If predictions equal true values, MAE and RMSE must be 0
    and R2 must be 1.
    """
    y_true = np.array([100, 200, 300])
    y_pred = np.array([100, 200, 300])
    metrics = compute_metrics(y_true, y_pred)

    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)


def test_compute_metrics_keys():
    """Metrics dict must contain exactly mae, rmse, r2."""
    y_true = np.array([100, 200, 300])
    y_pred = np.array([110, 190, 310])
    metrics = compute_metrics(y_true, y_pred)

    assert set(metrics.keys()) == {"mae", "rmse", "r2"}


def test_chunks_exist():
    """Monthly chunks must exist before training can run."""
    assert CHUNKS_DIR.exists(), "data/monthly_chunks/ directory not found"
    files = list(CHUNKS_DIR.glob("*.csv"))
    assert len(files) == 24, f"Expected 24 chunks, found {len(files)}"