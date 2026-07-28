# src/tests/test_data_prep.py
"""
Unit tests for data preparation module.

What we test here:
- load_raw_data: reads CSV and parses date column correctly
- build_features: removes leakage columns
- split_into_monthly_chunks: creates correct number of files
"""

import pytest
import pandas as pd
from pathlib import Path
from src.data.data_prep import load_raw_data, build_features, split_into_monthly_chunks

RAW_PATH = Path("data/raw/hour.csv")


def test_load_raw_data_shape():
    """Dataset must have 17379 rows and dteday must be datetime."""
    df = load_raw_data(RAW_PATH)
    assert len(df) == 17379
    assert pd.api.types.is_datetime64_any_dtype(df["dteday"])


def test_load_raw_data_no_missing():
    """Raw dataset must have no missing values."""
    df = load_raw_data(RAW_PATH)
    assert df.isnull().sum().sum() == 0


def test_build_features_removes_leakage():
    """build_features must remove casual, registered, and instant columns."""
    df = load_raw_data(RAW_PATH)
    df = build_features(df)
    for col in ["casual", "registered", "instant"]:
        assert col not in df.columns, f"Leakage column '{col}' was not removed"


def test_build_features_keeps_target():
    """Target column 'cnt' must remain after feature building."""
    df = load_raw_data(RAW_PATH)
    df = build_features(df)
    assert "cnt" in df.columns


def test_split_into_monthly_chunks(tmp_path):
    """
    split_into_monthly_chunks must create exactly 24 files
    (12 months x 2 years = 2011-01 to 2012-12).

    Why tmp_path?
    pytest provides tmp_path as a temporary directory that is
    automatically cleaned up after the test — we don't pollute
    the real data/monthly_chunks/ folder during testing.
    """
    df = load_raw_data(RAW_PATH)
    df = build_features(df)
    files = split_into_monthly_chunks(df, out_dir=tmp_path)

    assert len(files) == 24
    assert all(f.suffix == ".csv" for f in files)
    assert (tmp_path / "2011-01.csv").exists()
    assert (tmp_path / "2012-12.csv").exists()


def test_monthly_chunk_no_leakage(tmp_path):
    """Each chunk must not contain leakage columns."""
    df = load_raw_data(RAW_PATH)
    df = build_features(df)
    split_into_monthly_chunks(df, out_dir=tmp_path)

    sample = pd.read_csv(tmp_path / "2011-06.csv")
    for col in ["casual", "registered", "instant"]:
        assert col not in sample.columns