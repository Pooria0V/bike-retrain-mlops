# src/training/evaluate.py
"""
Compare the new model (challenger) against the current production model (champion).
Output is a simple decision: promote or reject.
"""

import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CHUNKS_DIR = Path("data/monthly_chunks")
TARGET_COLUMN = "cnt"
DROP_AT_TRAIN = ["dteday"]
IMPROVEMENT_THRESHOLD = 0.05

# Read tracking URI from environment variable so it works both
# locally (127.0.0.1:5001) and inside Docker (http://mlflow:5001)
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")


def load_test_data(test_month: str) -> tuple:
    """
    Load one specific month as the test set.

    Why a separate month?
    For a fair comparison, both models (champion and challenger)
    must be evaluated on exactly the same data.
    This month is typically the new incoming data that triggered retraining.
    """
    file = CHUNKS_DIR / f"{test_month}.csv"
    if not file.exists():
        raise FileNotFoundError(f"File not found: {file}")

    df = pd.read_csv(file)
    X = df.drop(columns=[TARGET_COLUMN] + DROP_AT_TRAIN, errors="ignore")
    y = df[TARGET_COLUMN]
    return X, y


def load_model_by_run_id(run_id: str):
    """
    Load a model from MLflow using its run_id.
    Used to load the challenger (freshly trained model).
    """
    model_uri = f"runs:/{run_id}/model"
    return mlflow.sklearn.load_model(model_uri)


def load_production_model():
    """
    Load the current champion model from MLflow Model Registry.
    Returns None if no champion exists yet.

    Uses the 'champion' alias set by promote.py —
    always points to the latest promoted model version.
    """
    client = mlflow.tracking.MlflowClient()
    try:
        client.get_model_version_by_alias("bike-demand-model", "champion")
        model_uri = "models:/bike-demand-model@champion"
        return mlflow.sklearn.load_model(model_uri)
    except (mlflow.exceptions.MlflowException, OSError):
        return None


def evaluate_model(model, X, y) -> dict:
    """Compute evaluation metrics for a given model."""
    y_pred = model.predict(X)
    return {
        "mae": mean_absolute_error(y, y_pred),
        "rmse": np.sqrt(mean_squared_error(y, y_pred)),
        "r2": r2_score(y, y_pred),
    }


def should_promote(champion_metrics: dict, challenger_metrics: dict) -> bool:
    """
    Decide whether the challenger is better than the current champion.

    Comparison criterion: RMSE
    Why RMSE? It penalizes large errors more heavily —
    in demand forecasting, big mistakes matter more than small ones.

    Promotion condition: challenger must have at least 5% lower RMSE.
    """
    champion_rmse = champion_metrics["rmse"]
    challenger_rmse = challenger_metrics["rmse"]
    improvement = (champion_rmse - challenger_rmse) / champion_rmse
    return improvement >= IMPROVEMENT_THRESHOLD


def run_evaluation(challenger_run_id: str, test_month: str) -> dict:
    """
    Run the full evaluation and comparison process.

    Returns a dict with the promotion decision and metrics for both models.
    """
    mlflow.set_tracking_uri(TRACKING_URI)

    print(f"\nLoading test data for month {test_month}...")
    X_test, y_test = load_test_data(test_month)
    print(f"  {len(X_test)} rows for testing")

    print("\nLoading challenger model...")
    challenger = load_model_by_run_id(challenger_run_id)
    challenger_metrics = evaluate_model(challenger, X_test, y_test)

    print("Loading champion model (production)...")
    champion = load_production_model()

    if champion is None:
        print("  No production model found — challenger promoted directly")
        promote = True
        champion_metrics = None
    else:
        champion_metrics = evaluate_model(champion, X_test, y_test)
        promote = should_promote(champion_metrics, challenger_metrics)

    print("\n" + "=" * 45)
    print(f"{'Metric':<10} {'Champion':>12} {'Challenger':>12}")
    print("-" * 45)
    for metric in ["mae", "rmse", "r2"]:
        champ_val = f"{champion_metrics[metric]:.2f}" if champion_metrics else "N/A"
        print(f"{metric:<10} {champ_val:>12} {challenger_metrics[metric]:>12.2f}")
    print("=" * 45)
    print(f"Decision: {'PROMOTE' if promote else 'REJECT'}")

    return {
        "promote": promote,
        "challenger_run_id": challenger_run_id,
        "challenger_metrics": challenger_metrics,
        "champion_metrics": champion_metrics,
    }


if __name__ == "__main__":
    result = run_evaluation(
        challenger_run_id="16fa6eeb5a084d0ba21b1620d7a1a91d",
        test_month="2011-06",
    )