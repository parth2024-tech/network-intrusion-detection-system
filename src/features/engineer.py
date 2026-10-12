"""
src/features/engineer.py
=========================
Feature engineering for NIDS flow data:

* Add derived ratio features
* Drop near-zero variance features
* Drop highly-correlated feature pairs
* (Optional) variance / correlation thresholds from config

Usage (API)::

    from src.features.engineer import engineer_features
    df_engineered = engineer_features(df_clean)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config(config_path: str = "config/model_config.yaml") -> dict:
    p = Path(config_path)
    if not p.exists():
        return {}
    with p.open() as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Derived features
# ---------------------------------------------------------------------------

def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived ratio features that highlight asymmetry between forward
    and backward traffic — a key indicator of scans and DDoS.

    Parameters
    ----------
    df:
        Cleaned DataFrame (column names already stripped).

    Returns
    -------
    pd.DataFrame
        DataFrame with additional columns:
        ``fwd_bwd_packet_ratio``, ``fwd_bwd_bytes_ratio``, ``bytes_per_packet``.
    """
    eps = 1e-9  # avoid division by zero

    fwd_pkts = df.get("Total Fwd Packets", pd.Series(0, index=df.index))
    bwd_pkts = df.get("Total Backward Packets", pd.Series(0, index=df.index))
    fwd_bytes = df.get("Total Length of Fwd Packets", pd.Series(0, index=df.index))
    bwd_bytes = df.get("Total Length of Bwd Packets", pd.Series(0, index=df.index))
    total_pkts = fwd_pkts + bwd_pkts
    total_bytes = fwd_bytes + bwd_bytes

    df = df.copy()
    df["fwd_bwd_packet_ratio"] = fwd_pkts / (bwd_pkts + eps)
    df["fwd_bwd_bytes_ratio"] = fwd_bytes / (bwd_bytes + eps)
    df["bytes_per_packet"] = total_bytes / (total_pkts + eps)

    logger.debug("Added 3 ratio features.")
    return df


# ---------------------------------------------------------------------------
# Feature selection helpers
# ---------------------------------------------------------------------------

def drop_low_variance(df: pd.DataFrame, threshold: float = 0.001) -> pd.DataFrame:
    """
    Drop numeric features with variance below *threshold*.

    Parameters
    ----------
    df:
        DataFrame (must not include label columns yet).
    threshold:
        Minimum variance to keep.

    Returns
    -------
    pd.DataFrame
    """
    meta_cols = {"Label", "label_binary", "label_multi", "source_file"}
    num_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in meta_cols
    ]
    variances = df[num_cols].var()
    low_var = variances[variances < threshold].index.tolist()
    if low_var:
        logger.info("Dropping %d low-variance features: %s", len(low_var), low_var[:10])
        df = df.drop(columns=low_var)
    return df


def drop_correlated_features(
    df: pd.DataFrame, threshold: float = 0.98
) -> pd.DataFrame:
    """
    Drop one feature from each pair with Pearson correlation > *threshold*.

    Parameters
    ----------
    df:
        DataFrame.
    threshold:
        Absolute correlation threshold.

    Returns
    -------
    pd.DataFrame
    """
    meta_cols = {"Label", "label_binary", "label_multi", "source_file"}
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in meta_cols]

    corr_matrix = df[num_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    if to_drop:
        logger.info(
            "Dropping %d highly-correlated features (threshold=%.2f): %s …",
            len(to_drop), threshold, to_drop[:5],
        )
        df = df.drop(columns=to_drop)
    else:
        logger.info("No highly-correlated features found (threshold=%.2f).", threshold)
    return df


# ---------------------------------------------------------------------------
# Master engineering function
# ---------------------------------------------------------------------------

def engineer_features(
    df: pd.DataFrame,
    config_path: str = "config/model_config.yaml",
) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline.

    Steps:
    1. Add ratio features
    2. Drop low-variance features
    3. Drop highly-correlated features

    Parameters
    ----------
    df:
        Cleaned DataFrame.
    config_path:
        Path to YAML config.

    Returns
    -------
    pd.DataFrame
        Engineered DataFrame ready for train/test split.
    """
    cfg = _load_config(config_path)
    feat_cfg = cfg.get("features", {})

    if feat_cfg.get("add_ratios", True):
        df = add_ratio_features(df)

    var_threshold = feat_cfg.get("variance_threshold", 0.001)
    df = drop_low_variance(df, threshold=var_threshold)

    corr_threshold = feat_cfg.get("correlation_threshold", 0.98)
    df = drop_correlated_features(df, threshold=corr_threshold)

    logger.info("Feature engineering complete. Columns: %d", df.shape[1])
    return df
