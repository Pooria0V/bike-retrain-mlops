# src/training/train.py
"""
Training a regression model on Bike Sharing data
and logging the full experiment to MLflow.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

CHUNKS_DIR = Path("data/monthly_chunks")
TARGET_COLUMN = "cnt"
DROP_AT_TRAIN = ["dteday"]

# Read tracking URI from environment variable so it works both
# locally (127.0.0.1:5001) and inside Docker (http://mlflow:5001)
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")


def load_chunks_up_to(month: str, chunks_dir: Path = CHUNKS_DIR) -> pd.DataFrame:
    """
    Load all monthly chunks from the beginning up to the given month.

    Example: month="2011-06" loads 2011-01.csv through 2011-06.csv
    and concatenates them into one DataFrame.

    Why this approach? In real-world retraining, we always train
    on all available data up to now, not just the latest month.
    """
    all_files = sorted(chunks_dir.glob("*.csv"))
    selected = [f for f in all_files if f.stem <= month]

    if not selected:
        raise ValueError(f"No chunks found up to month {month}")

    df = pd.concat([pd.read_csv(f) for f in selected], ignore_index=True)
    print(f"  {len(selected)} months loaded | {len(df)} rows")
    return df


def prepare_X_y(df: pd.DataFrame):
    """
    Separate features (X) from target (y).
    dteday is also dropped here since the model
    cannot work directly with date strings.
    """
    X = df.drop(columns=[TARGET_COLUMN] + DROP_AT_TRAIN, errors="ignore")
    y = df[TARGET_COLUMN]
    return X, y


def compute_metrics(y_true, y_pred) -> dict:
    """
    Compute regression evaluation metrics:
    - MAE : mean absolute error (same unit as target — number of bikes)
    - RMSE: root mean squared error (penalizes large errors more)
    - R2  : fraction of variance explained (1 = perfect, 0 = useless)
    """
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def train_model(
    up_to_month: str,
    experiment_name: str = "bike-retrain",
    model_params: dict = None,
) -> str:
    """
    Train a model and log everything to MLflow.

    Args:
        up_to_month    : last month to include in training (e.g. "2011-06")
        experiment_name: MLflow experiment name
        model_params   : RandomForest hyperparameters

    Returns:
        run_id: unique identifier of this MLflow run
                (used later for evaluation and promotion)
    """
    if model_params is None:
        model_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
        }

    # --- Load and prepare data ---
    print(f"\nLoading data up to month {up_to_month}...")
    df = load_chunks_up_to(up_to_month)
    X, y = prepare_X_y(df)

    # Split into train (80%) and test (20%)
    # shuffle=False because data is time-ordered —
    # shuffling would leak future data into training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    os.environ["MLFLOW_TRACKING_URI"] = TRACKING_URI

    # --- Configure MLflow ---
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        mlflow.set_tag("up_to_month", up_to_month)
        mlflow.set_tag("train_rows", len(X_train))
        mlflow.log_params(model_params)

        # --- Train ---
        print("Training model...")
        model = RandomForestRegressor(**model_params)
        model.fit(X_train, y_train)

        # --- Evaluate ---
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        mlflow.log_metrics(metrics)

        # Save model to MLflow — retrievable later from Model Registry
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="bike-demand-model",
        )

        run_id = run.info.run_id

    print(f"\nTraining results up to {up_to_month}:")
    print(f"  MAE  : {metrics['mae']:.2f} bikes")
    print(f"  RMSE : {metrics['rmse']:.2f} bikes")
    print(f"  R2   : {metrics['r2']:.4f}")
    print(f"  Run ID: {run_id}")

    return run_id


if __name__ == "__main__":
    train_model(up_to_month="2011-06")