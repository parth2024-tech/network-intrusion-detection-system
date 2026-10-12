"""
src/data/preprocess.py
=======================
Master preprocessing pipeline:

1. Load raw CSVs  (loader.py)
2. Clean data     (cleaner.py)
3. Engineer features  (features/engineer.py)
4. Scale features  (StandardScaler)
5. Time-based train/test split
6. Save processed artifacts

Usage (CLI)::

    python -m src.data.preprocess \\
        --raw-dir data/raw \\
        --out-dir data/processed

Usage (API)::

    from src.data.preprocess import run_preprocessing
    X_train, X_test, y_binary_train, y_binary_test, y_multi_train, y_multi_test, scaler = \\
        run_preprocessing("data/raw")
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.data.cleaner import clean
from src.data.loader import load_cicids2017
from src.features.engineer import engineer_features

logger = logging.getLogger(__name__)

_SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


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
# Train / test split  (time-based, never random shuffle)
# ---------------------------------------------------------------------------

def time_based_split(
    df: pd.DataFrame,
    test_files: list[str],
    source_col: str = "source_file",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by source file (day) to prevent temporal leakage.

    Parameters
    ----------
    df:
        Combined cleaned DataFrame.
    test_files:
        List of CSV filenames to use as test set.
    source_col:
        Column storing the original CSV filename.

    Returns
    -------
    (train_df, test_df)
    """
    mask = df[source_col].isin(test_files)
    train_df = df[~mask].copy()
    test_df = df[mask].copy()
    logger.info(
        "Train rows: %d  |  Test rows: %d  (test files: %s)",
        len(train_df), len(test_df), test_files,
    )
    if len(test_df) == 0:
        logger.warning(
            "Test set is empty! "
            "Check that test_files match actual source_file values. "
            "Falling back to last-25%% time fraction split."
        )
        n = len(df)
        train_df = df.iloc[: int(n * 0.75)].copy()
        test_df = df.iloc[int(n * 0.75) :].copy()
    return train_df, test_df


# ---------------------------------------------------------------------------
# Feature matrix extraction
# ---------------------------------------------------------------------------

def _get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature columns (exclude label and metadata columns)."""
    exclude = {
        "Label", "label_binary", "label_multi",
        "source_file",
    }
    return [c for c in df.columns if c not in exclude]


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------

def run_preprocessing(
    raw_dir: str = "data/raw",
    out_dir: str = "data/processed",
    config_path: str = "config/model_config.yaml",
    *,
    save: bool = True,
) -> tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    StandardScaler,
    list[str],
]:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    raw_dir:
        Directory containing raw CICIDS2017 CSVs.
    out_dir:
        Directory to write processed parquet files and scaler.
    config_path:
        Path to YAML config.
    save:
        If True, persist processed splits and scaler to *out_dir*.

    Returns
    -------
    X_train, X_test, y_bin_train, y_bin_test, y_multi_train, y_multi_test, scaler, feature_cols
    """
    cfg = _load_config(config_path)
    preproc_cfg = cfg.get("preprocessing", {})
    paths_cfg = cfg.get("paths", {})

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load raw data
    logger.info("=== Step 1: Loading raw CSVs ===")
    df_raw = load_cicids2017(raw_dir=raw_dir, config_path=config_path)

    # 2. Clean
    logger.info("=== Step 2: Cleaning ===")
    df_clean = clean(df_raw, config_path=config_path)

    # 3. Feature engineering
    logger.info("=== Step 3: Feature Engineering ===")
    df_feat = engineer_features(df_clean)

    # 4. Time-based split
    logger.info("=== Step 4: Train/Test Split (by day) ===")
    test_files: list[str] = preproc_cfg.get("test_days", [])
    train_df, test_df = time_based_split(df_feat, test_files=test_files)

    feature_cols = _get_feature_columns(train_df)

    X_train = train_df[feature_cols].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_bin_train = train_df["label_binary"].values
    y_bin_test = test_df["label_binary"].values
    y_multi_train = train_df["label_multi"].values
    y_multi_test = test_df["label_multi"].values

    # 5. Scale — fit ONLY on train
    logger.info("=== Step 5: Scaling ===")
    scaler_name = preproc_cfg.get("scaler", "standard")
    scaler_cls = _SCALERS.get(scaler_name, StandardScaler)
    scaler = scaler_cls()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    logger.info("Scaler: %s  |  X_train: %s  X_test: %s", scaler_name, X_train.shape, X_test.shape)

    if save:
        # Save numpy arrays as parquet-compatible DataFrames
        train_out = pd.DataFrame(X_train, columns=feature_cols)
        train_out["label_binary"] = y_bin_train
        train_out["label_multi"] = y_multi_train
        train_out.to_parquet(out_path / "train.parquet", index=False)

        test_out = pd.DataFrame(X_test, columns=feature_cols)
        test_out["label_binary"] = y_bin_test
        test_out["label_multi"] = y_multi_test
        test_out.to_parquet(out_path / "test.parquet", index=False)

        # Save feature list
        import json
        (out_path / "feature_cols.json").write_text(json.dumps(feature_cols, indent=2))

        # Save scaler
        scaler_path = paths_cfg.get("scaler", "models/scaler.joblib")
        Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)
        logger.info("Scaler saved → %s", scaler_path)
        logger.info("Train/Test parquets saved → %s", out_path)

    return X_train, X_test, y_bin_train, y_bin_test, y_multi_train, y_multi_test, scaler, feature_cols


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full NIDS preprocessing pipeline")
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--out-dir", default="data/processed")
    p.add_argument("--config", default="config/model_config.yaml")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()
    run_preprocessing(raw_dir=args.raw_dir, out_dir=args.out_dir, config_path=args.config)
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
