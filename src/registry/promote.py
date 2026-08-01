# src/registry/promote.py
"""
Model promotion management in MLflow Model Registry.
Uses aliases instead of stages (stages deprecated since MLflow 2.9).
"""

import os

import mlflow
from mlflow.tracking import MlflowClient

# Read tracking URI from environment variable so it works both
# locally (127.0.0.1:5001) and inside Docker (http://mlflow:5001)
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
MODEL_NAME = "bike-demand-model"
CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"


def get_model_version_by_run_id(client: MlflowClient, run_id: str) -> str:
    """Find model version number using run_id."""
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    for v in versions:
        if v.run_id == run_id:
            return v.version
    raise ValueError(f"No model version found with run_id={run_id}")


def get_champion_version(client: MlflowClient) -> str | None:
    """Get current champion version. Returns None if no champion exists."""
    try:
        version_info = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
        return version_info.version
    except (mlflow.exceptions.MlflowException, OSError):
        return None


def promote_challenger(challenger_run_id: str) -> bool:
    """
    Promotes challenger model to champion.

    Steps:
    1. Find version number by run_id
    2. Assign 'challenger' alias to new model
    3. Remove 'champion' alias from old model
    4. Assign 'champion' alias to new model
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    print(f"\nPromoting model with run_id: {challenger_run_id}")

    new_version = get_model_version_by_run_id(client, challenger_run_id)
    print(f"  New version: {new_version}")

    client.set_registered_model_alias(MODEL_NAME, CHALLENGER_ALIAS, new_version)
    print(f"  Alias '{CHALLENGER_ALIAS}' assigned to version {new_version}")

    old_champion_version = get_champion_version(client)
    if old_champion_version:
        client.delete_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS)
        print(f"  Alias '{CHAMPION_ALIAS}' removed from version {old_champion_version}")

    client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, new_version)
    print(f"  Alias '{CHAMPION_ALIAS}' assigned to version {new_version} -> PROMOTED successfully")

    return True


def reject_challenger(challenger_run_id: str) -> bool:
    """Rejects challenger — current champion remains unchanged."""
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    print(f"\nModel with run_id {challenger_run_id} -> REJECTED")
    print("  Current champion remains unchanged")

    try:
        client.delete_registered_model_alias(MODEL_NAME, CHALLENGER_ALIAS)
    except (mlflow.exceptions.MlflowException, OSError):
        print("Challenger alias not found, skipping deletion")

    return False


def run_promotion(challenger_run_id: str, should_promote: bool) -> bool:
    """Main entry point — acts based on evaluate.py decision."""
    if should_promote:
        return promote_challenger(challenger_run_id)
    else:
        return reject_challenger(challenger_run_id)


if __name__ == "__main__":
    RUN_ID = "72d4fa5846984405b9012e1473e86004"
    run_promotion(challenger_run_id=RUN_ID, should_promote=True)