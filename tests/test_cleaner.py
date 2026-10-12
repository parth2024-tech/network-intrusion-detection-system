"""
tests/test_cleaner.py
======================
Unit tests for src/data/cleaner.py
"""
import numpy as np
import pandas as pd
import pytest

from src.data.cleaner import (
    drop_duplicates,
    drop_irrelevant_columns,
    encode_binary_labels,
    encode_multiclass_labels,
    impute_missing,
    replace_infinities,
    strip_column_names,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Small CICIDS2017-like DataFrame with known quirks."""
    return pd.DataFrame({
        " Flow Duration": [100.0, 200.0, np.inf, 400.0, 100.0],
        " Total Fwd Packets": [10, 20, 30, 40, 10],
        " Label": ["BENIGN", "BENIGN", "DDoS", "PortScan", "BENIGN"],
        " Flow ID": ["a", "b", "c", "d", "a"],
        " Source IP": ["1.1.1.1"] * 5,
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStripColumnNames:
    def test_strips_leading_and_trailing(self, sample_df):
        df = strip_column_names(sample_df.copy())
        assert "Flow Duration" in df.columns
        assert "Label" in df.columns
        assert " Label" not in df.columns


class TestReplaceInfinities:
    def test_replaces_inf_with_nan(self, sample_df):
        df = strip_column_names(sample_df.copy())
        df = replace_infinities(df)
        assert not np.isinf(df["Flow Duration"].values).any()
        assert df["Flow Duration"].isna().sum() == 1


class TestImputeMissing:
    def test_median_fills_nan(self):
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": [1, 2, 3, 4]})
        df = impute_missing(df, strategy="median")
        assert df["a"].isna().sum() == 0
        assert df["a"].iloc[2] == pytest.approx(2.0)  # median of [1,2,4]

    def test_zero_fill(self):
        df = pd.DataFrame({"x": [np.nan, 5.0]})
        df = impute_missing(df, strategy="zero")
        assert df["x"].iloc[0] == 0.0

    def test_unknown_strategy_raises(self):
        df = pd.DataFrame({"x": [1.0]})
        with pytest.raises(ValueError, match="Unknown impute strategy"):
            impute_missing(df, strategy="magic")


class TestDropDuplicates:
    def test_removes_exact_duplicates(self, sample_df):
        df = strip_column_names(sample_df.copy())
        n_before = len(df)
        df = drop_duplicates(df)
        # Row 0 and Row 4 are duplicates → 1 should be removed
        assert len(df) < n_before


class TestDropIrrelevantColumns:
    def test_drops_expected_columns(self, sample_df):
        df = strip_column_names(sample_df.copy())
        df = drop_irrelevant_columns(df, drop_cols=["Flow ID", "Source IP"])
        assert "Flow ID" not in df.columns
        assert "Source IP" not in df.columns
        assert "Flow Duration" in df.columns  # should be preserved


class TestEncodeBinaryLabels:
    def test_benign_is_zero_attack_is_one(self):
        df = pd.DataFrame({"Label": ["BENIGN", "DDoS", "BENIGN", "PortScan"]})
        df = encode_binary_labels(df, label_col="Label")
        assert list(df["label_binary"]) == [0, 1, 0, 1]

    def test_all_benign(self):
        df = pd.DataFrame({"Label": ["BENIGN", "BENIGN"]})
        df = encode_binary_labels(df, label_col="Label")
        assert df["label_binary"].sum() == 0


class TestEncodeMulticlassLabels:
    def test_known_labels_mapped_correctly(self):
        df = pd.DataFrame({"Label": ["BENIGN", "DDoS", "PortScan", "Bot"]})
        label_map = {"BENIGN": 0, "DDoS": 2, "PortScan": 7, "Bot": 6}
        df = encode_multiclass_labels(df, label_col="Label", label_map=label_map)
        assert list(df["label_multi"]) == [0, 2, 7, 6]

    def test_unknown_label_gets_minus_one(self):
        df = pd.DataFrame({"Label": ["UNKNOWN"]})
        df = encode_multiclass_labels(df, label_col="Label", label_map={"BENIGN": 0})
        assert df["label_multi"].iloc[0] == -1
