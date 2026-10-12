"""
tests/test_engineer.py
=======================
Unit tests for src/features/engineer.py
"""
import numpy as np
import pandas as pd
import pytest

from src.features.engineer import (
    add_ratio_features,
    drop_correlated_features,
    drop_low_variance,
)


@pytest.fixture()
def clean_df() -> pd.DataFrame:
    """Minimal clean DataFrame (label columns already stripped)."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "Total Fwd Packets": np.random.randint(1, 100, n).astype(float),
        "Total Backward Packets": np.random.randint(0, 50, n).astype(float),
        "Total Length of Fwd Packets": np.random.rand(n) * 10000,
        "Total Length of Bwd Packets": np.random.rand(n) * 5000,
        "Flow Duration": np.random.rand(n) * 1e6,
        "SYN Flag Count": np.random.randint(0, 10, n).astype(float),
        "label_binary": np.random.randint(0, 2, n),
        "label_multi": np.random.randint(0, 8, n),
    })


class TestAddRatioFeatures:
    def test_adds_three_columns(self, clean_df):
        df = add_ratio_features(clean_df.copy())
        assert "fwd_bwd_packet_ratio" in df.columns
        assert "fwd_bwd_bytes_ratio" in df.columns
        assert "bytes_per_packet" in df.columns

    def test_no_division_by_zero(self, clean_df):
        df_zero = clean_df.copy()
        df_zero["Total Backward Packets"] = 0
        df_zero["Total Length of Bwd Packets"] = 0
        df = add_ratio_features(df_zero)
        assert np.isfinite(df["fwd_bwd_packet_ratio"]).all()
        assert np.isfinite(df["fwd_bwd_bytes_ratio"]).all()


class TestDropLowVariance:
    def test_drops_constant_column(self, clean_df):
        df = clean_df.copy()
        df["constant_col"] = 5.0  # zero variance
        df = drop_low_variance(df, threshold=0.001)
        assert "constant_col" not in df.columns

    def test_keeps_high_variance_column(self, clean_df):
        df = drop_low_variance(clean_df.copy(), threshold=0.001)
        assert "Flow Duration" in df.columns


class TestDropCorrelatedFeatures:
    def test_drops_perfectly_correlated_column(self, clean_df):
        df = clean_df.copy()
        df["fwd_clone"] = df["Total Fwd Packets"] * 1.0  # perfect correlation
        n_before = len(df.columns)
        df = drop_correlated_features(df, threshold=0.98)
        assert len(df.columns) < n_before

    def test_keeps_uncorrelated_columns(self, clean_df):
        df = drop_correlated_features(clean_df.copy(), threshold=0.98)
        # At minimum these key features should survive
        assert "label_binary" in df.columns
