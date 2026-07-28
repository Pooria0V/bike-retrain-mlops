# src/data/data_prep.py
"""
Prepares the Bike Sharing dataset and splits it into monthly chunks
to simulate incremental data arrival in the auto-retraining pipeline.
"""

import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/hour.csv")
CHUNKS_DIR = Path("data/monthly_chunks")

# These columns must be removed to prevent data leakage:
# - casual and registered: their sum equals cnt (they reveal the answer)
# - instant: just a row index, carries no predictive information
LEAKAGE_COLUMNS = ["casual", "registered", "instant"]

TARGET_COLUMN = "cnt"


def load_raw_data(path: Path = RAW_PATH) -> pd.DataFrame:
    """
    Read raw dataset from disk.
    Parses dteday as datetime so we can perform
    time-based operations like monthly groupby later.
    """
    df = pd.read_csv(path)
    df["dteday"] = pd.to_datetime(df["dteday"])
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove leakage columns.
    Remaining features (season, hr, weathersit, temp, ...)
    are already present in the dataset and need no extra engineering.
    """
    df = df.drop(columns=LEAKAGE_COLUMNS)
    return df


def split_into_monthly_chunks(df: pd.DataFrame, out_dir: Path = CHUNKS_DIR) -> list[Path]:
    """
    Split the DataFrame by year-month and save each month
    as a separate CSV file.

    Output example:
        data/monthly_chunks/2011-01.csv
        data/monthly_chunks/2011-02.csv
        ...
        data/monthly_chunks/2012-12.csv

    Why split by month?
    The Prefect flow feeds one new month at a time into the pipeline,
    simulating real-world data arriving incrementally.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # to_period("M") maps each date to its month period
    # e.g. 2011-01-15 -> 2011-01
    df["year_month"] = df["dteday"].dt.to_period("M")

    written_files = []
    for period, chunk in df.groupby("year_month"):
        chunk = chunk.drop(columns=["year_month"])
        file_path = out_dir / f"{period}.csv"
        chunk.to_csv(file_path, index=False)
        written_files.append(file_path)

    return sorted(written_files)


def run():
    """Main entry point."""
    print("Loading raw dataset...")
    df = load_raw_data()
    print(f"  {len(df)} rows loaded")

    print("Removing leakage columns...")
    df = build_features(df)
    print(f"  Remaining columns: {list(df.columns)}")

    print("Splitting into monthly chunks...")
    files = split_into_monthly_chunks(df)
    print(f"  {len(files)} monthly chunks saved to {CHUNKS_DIR}")

    return files


if __name__ == "__main__":
    run()