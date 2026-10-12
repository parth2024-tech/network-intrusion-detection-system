"""
src/data/cleaner.py
====================
CICIDS2017-specific data cleaning:

* Replace ±Inf with NaN then impute
* Drop duplicate rows
* Drop columns with near-zero or all-null values
* Strip column-name whitespace
* Apply binary and multiclass label encoding

Usage (CLI)::

    python -m src.data.cleaner \\
        --input  data/processed/raw_combined.parquet \\
        --output data/processed/cleaned.parquet

Usage (API)::

    from src.data.cleaner import clean
    df_clean, label_encoder = clean(df_raw)
"""
from __future__ import annotations

import argparse
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
# Cleaning functions
# ---------------------------------------------------------------------------

def strip_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove leading/trailing whitespace from all column names.

    CICIDS2017 has inconsistent spacing (e.g. ``' Label'``, ``' Flow Duration'``).

    Parameters
    ----------
    df:
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with stripped column names (in-place).
    """
    df.columns = df.columns.str.strip()
    return df


def replace_infinities(df: pd.DataFrame, numeric_only: bool = True) -> pd.DataFrame:
    """
    Replace ±Inf values with NaN so they can be imputed downstream.

    Parameters
    ----------
    df:
        Input DataFrame.
    numeric_only:
        If True, only process numeric columns.

    Returns
    -------
    pd.DataFrame
    """
    num_cols = df.select_dtypes(include=[np.number]).columns
    n_inf = np.isinf(df[num_cols]).sum().sum()
    if n_inf:
        logger.info("Replacing %d ±Inf values with NaN.", n_inf)
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    return df


def impute_missing(
    df: pd.DataFrame, strategy: str = "median"
) -> pd.DataFrame:
    """
    Impute NaN values in numeric columns.

    Parameters
    ----------
    df:
        Input DataFrame.
    strategy:
        ``'median'``, ``'mean'``, or ``'zero'``.

    Returns
    -------
    pd.DataFrame
    """
    num_cols = df.select_dtypes(include=[np.number]).columns
    n_nan = df[num_cols].isna().sum().sum()
    logger.info("Imputing %d NaN values (strategy=%s).", n_nan, strategy)

    if strategy == "median":
        fill_vals = df[num_cols].median()
    elif strategy == "mean":
        fill_vals = df[num_cols].mean()
    elif strategy == "zero":
        fill_vals = pd.Series(0, index=num_cols)
    else:
        raise ValueError(f"Unknown impute strategy: {strategy!r}")

    df[num_cols] = df[num_cols].fillna(fill_vals)
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows.

    Parameters
    ----------
    df:
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
    """
    n_before = len(df)
    df = df.drop_duplicates()
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.info("Dropped %d duplicate rows.", n_dropped)
    return df


def drop_irrelevant_columns(
    df: pd.DataFrame, drop_cols: list[str] | None = None
) -> pd.DataFrame:
    """
    Drop columns that are identifiers or metadata (not model features).

    Parameters
    ----------
    df:
        Input DataFrame.
    drop_cols:
        Column names to remove.  Defaults to ``['Flow ID', 'Source IP',
        'Source Port', 'Destination IP', 'Timestamp']``.

    Returns
    -------
    pd.DataFrame
    """
    if drop_cols is None:
        drop_cols = [
            "Flow ID", "Source IP", "Source Port",
            "Destination IP", "Timestamp",
        ]
    existing = [c for c in drop_cols if c in df.columns]
    if existing:
        logger.info("Dropping identifier columns: %s", existing)
        df = df.drop(columns=existing)
    return df


def encode_binary_labels(
    df: pd.DataFrame,
    label_col: str = "Label",
    benign_label: str = "BENIGN",
) -> pd.DataFrame:
    """
    Add a ``label_binary`` column: 0 = benign, 1 = attack.

    Parameters
    ----------
    df:
        Cleaned DataFrame (column names already stripped).
    label_col:
        Name of the original label column (after stripping).
    benign_label:
        Value that represents normal traffic.

    Returns
    -------
    pd.DataFrame
    """
    df["label_binary"] = (df[label_col] != benign_label).astype(int)
    attack_rate = df["label_binary"].mean() * 100
    logger.info(
        "Binary labels: %d BENIGN, %d ATTACK (%.2f%% attack rate)",
        (df["label_binary"] == 0).sum(),
        (df["label_binary"] == 1).sum(),
        attack_rate,
    )
    return df


def encode_multiclass_labels(
    df: pd.DataFrame,
    label_col: str = "Label",
    label_map: dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    Add a ``label_multi`` column mapping each attack type to an integer class.

    Parameters
    ----------
    df:
        Cleaned DataFrame.
    label_col:
        Original label column name.
    label_map:
        Dict mapping string label → integer class. Unknown labels get -1.

    Returns
    -------
    pd.DataFrame
    """
    if label_map is None:
        label_map = {
            "BENIGN": 0,
            "FTP-Patator": 1, "SSH-Patator": 1,
            "DoS slowloris": 2, "DoS Slowhttptest": 2,
            "DoS Hulk": 2, "DoS GoldenEye": 2, "DDoS": 2,
            "Heartbleed": 3,
            "Web Attack - Brute Force": 4,
            "Web Attack - XSS": 4,
            "Web Attack - Sql Injection": 4,
            "Infiltration": 5,
            "Bot": 6,
            "PortScan": 7,
        }

    df["label_multi"] = df[label_col].map(label_map).fillna(-1).astype(int)
    unknown = (df["label_multi"] == -1).sum()
    if unknown:
        unknown_labels = df.loc[df["label_multi"] == -1, label_col].unique()
        logger.warning(
            "%d rows with unmapped labels (assigned -1): %s", unknown, unknown_labels
        )
    logger.info("Multiclass distribution:\n%s", df["label_multi"].value_counts())
    return df


# ---------------------------------------------------------------------------
# Master clean function
# ---------------------------------------------------------------------------

def clean(
    df: pd.DataFrame,
    config_path: str = "config/model_config.yaml",
) -> pd.DataFrame:
    """
    Run the full cleaning pipeline on a raw CICIDS2017 DataFrame.

    Steps:
    1. Strip column-name whitespace
    2. Drop identifier columns
    3. Replace ±Inf → NaN
    4. Impute NaN (median)
    5. Drop duplicates
    6. Encode binary labels
    7. Encode multiclass labels

    Parameters
    ----------
    df:
        Raw combined DataFrame from :func:`src.data.loader.load_cicids2017`.
    config_path:
        Path to YAML config.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with ``label_binary`` and ``label_multi`` columns.
    """
    cfg = _load_config(config_path)
    data_cfg = cfg.get("data", {})
    preproc_cfg = cfg.get("preprocessing", {})

    df = strip_column_names(df)
    df = drop_irrelevant_columns(
        df, drop_cols=preproc_cfg.get("drop_columns", None)
    )

    if preproc_cfg.get("replace_inf", True):
        df = replace_infinities(df)

    df = impute_missing(df, strategy=preproc_cfg.get("fill_nan_strategy", "median"))
    df = drop_duplicates(df)

    label_col = data_cfg.get("label_column", " Label").strip()
    benign_label = data_cfg.get("benign_label", "BENIGN")
    label_map = data_cfg.get("multiclass_map", None)

    df = encode_binary_labels(df, label_col=label_col, benign_label=benign_label)
    df = encode_multiclass_labels(df, label_col=label_col, label_map=label_map)

    logger.info("Cleaning complete. Shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean CICIDS2017 parquet file")
    p.add_argument("--input", default="data/processed/raw_combined.parquet")
    p.add_argument("--output", default="data/processed/cleaned.parquet")
    p.add_argument("--config", default="config/model_config.yaml")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    args = _parse_args()

    logger.info("Reading %s …", args.input)
    df = pd.read_parquet(args.input)

    df_clean = clean(df, config_path=args.config)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(out, index=False)
    logger.info("Saved cleaned data → %s  shape=%s", out, df_clean.shape)


if __name__ == "__main__":
    main()
