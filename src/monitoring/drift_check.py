# src/monitoring/drift_check.py
"""
Data drift detection using Evidently AI.
Compares statistical distribution of new data against training data
to decide whether retraining is needed.
"""

from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DatasetDriftMetric
from evidently.report import Report

CHUNKS_DIR = Path("data/monthly_chunks")
REPORTS_DIR = Path("reports/drift")
TARGET_COLUMN = "cnt"
DROP_AT_TRAIN = ["dteday"]

# If more than this fraction of features have drifted, trigger retraining
DRIFT_THRESHOLD = 0.5


def load_chunk(month: str) -> pd.DataFrame:
    """Load a single monthly chunk and drop non-feature columns."""
    file = CHUNKS_DIR / f"{month}.csv"
    if not file.exists():
        raise FileNotFoundError(f"Chunk not found: {file}")
    df = pd.read_csv(file)
    df = df.drop(columns=DROP_AT_TRAIN, errors="ignore")
    return df


def load_reference_data(up_to_month: str) -> pd.DataFrame:
    """
    Load all chunks up to a given month as reference (training) data.
    Reference data = what the model was trained on.
    """
    all_files = sorted(CHUNKS_DIR.glob("*.csv"))
    selected = [f for f in all_files if f.stem <= up_to_month]
    df = pd.concat([pd.read_csv(f) for f in selected], ignore_index=True)
    df = df.drop(columns=DROP_AT_TRAIN, errors="ignore")
    return df


def check_drift(reference_month: str, new_month: str) -> dict:
    """
    Compare new month data against reference (training) data.

    How Evidently works:
    - It runs statistical tests on each feature column
      (e.g. KS test for numerical, chi-square for categorical)
    - If the distribution of a feature has shifted significantly,
      that feature is marked as 'drifted'
    - If too many features drift -> model likely needs retraining

    Args:
        reference_month: last month used in training (e.g. "2011-06")
        new_month: the new incoming month to check (e.g. "2011-07")

    Returns:
        dict with drift decision and detailed metrics
    """
    print(f"\nChecking drift: reference up to {reference_month} | new month: {new_month}")

    reference_df = load_reference_data(reference_month)
    current_df = load_chunk(new_month)

    print(f"  Reference rows : {len(reference_df)}")
    print(f"  New month rows : {len(current_df)}")

    # Build Evidently report with two metrics:
    # 1. DataDriftPreset   -> per-feature drift analysis
    # 2. DatasetDriftMetric -> overall dataset drift summary
    report = Report(metrics=[
        DataDriftPreset(),
        DatasetDriftMetric(),
    ])

    report.run(reference_data=reference_df, current_data=current_df)

    # Extract results as a Python dict
    result = report.as_dict()

    # Parse overall drift result
    dataset_drift_result = result["metrics"][1]["result"]
    drift_detected = dataset_drift_result["dataset_drift"]
    share_drifted = dataset_drift_result["share_of_drifted_columns"]
    n_drifted = dataset_drift_result["number_of_drifted_columns"]
    n_total = dataset_drift_result["number_of_columns"]

    print(f"  Drifted features: {n_drifted}/{n_total} ({share_drifted:.1%})")
    print(f"  Drift detected  : {drift_detected}")
    print(f"  Decision        : {'RETRAIN' if drift_detected else 'KEEP CURRENT MODEL'}")

    # Save HTML report for visual inspection
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"drift_{reference_month}_vs_{new_month}.html"
    report.save_html(str(report_path))
    print(f"  Report saved to : {report_path}")

    return {
        "drift_detected": drift_detected,
        "share_drifted": share_drifted,
        "n_drifted": n_drifted,
        "n_total": n_total,
        "reference_month": reference_month,
        "new_month": new_month,
        "report_path": str(report_path),
    }


if __name__ == "__main__":
    result = check_drift(
        reference_month="2011-06",
        new_month="2011-07",
    )