"""
src/data/loader.py
==================
Load raw CICIDS2017 CSV files into a single pandas DataFrame,
tag each row with the source file/day, and apply the label column
normalisation needed before preprocessing.

Usage (CLI)::

    python -m src.data.loader --raw-dir data/raw --out data/processed/raw_combined.parquet

Usage (API)::

    from src.data.loader import load_cicids2017
    df = load_cicids2017("data/raw")
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_config(config_path: str = "config/model_config.yaml") -> dict:
    """Load YAML config. Returns empty dict if file not found."""
    p = Path(config_path)
    if not p.exists():
        logger.warning("Config not found at %s, using defaults.", config_path)
        return {}
    with p.open() as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def load_cicids2017(
    raw_dir: str | Path = "data/raw",
    config_path: str = "config/model_config.yaml",
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load all CICIDS2017 CSV files from *raw_dir* into one DataFrame.

    Parameters
    ----------
    raw_dir:
        Directory containing the raw ``*ISCX.csv`` files.
    config_path:
        Path to ``model_config.yaml``.
    verbose:
        Log per-file row counts when True.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with an extra ``source_file`` column identifying
        which CSV each row came from.

    Notes
    -----
    - CICIDS2017 column names include leading/trailing spaces — we strip them
      **after** concatenation so the label column mapping still works.
    - Encoding is latin-1 (the dataset has some non-UTF8 bytes).
    """
    raw_dir = Path(raw_dir)
    cfg = _load_config(config_path)
    label_col_raw: str = cfg.get("data", {}).get("label_column", " Label")

    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}. "
            "Download CICIDS2017 MachineLearningCSV.zip and extract here."
        )

    frames: list[pd.DataFrame] = []
    for csv_path in csv_files:
        try:
            df_part = pd.read_csv(
                csv_path,
                encoding="latin-1",
                low_memory=False,
            )
            df_part["source_file"] = csv_path.name
            frames.append(df_part)
            if verbose:
                logger.info("  Loaded %-60s  rows=%d", csv_path.name, len(df_part))
        except Exception as exc:
            logger.error("Failed to load %s: %s", csv_path.name, exc)
            raise

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Combined dataset: %d rows, %d columns", *combined.shape)

    # Normalise label column — strip whitespace from the label values
    if label_col_raw in combined.columns:
        combined[label_col_raw] = combined[label_col_raw].str.strip()

    return combined


def load_single_file(csv_path: str | Path) -> pd.DataFrame:
    """
    Load a single CICIDS2017 CSV.

    Parameters
    ----------
    csv_path:
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(csv_path, encoding="latin-1", low_memory=False)
    df["source_file"] = Path(csv_path).name
    df[" Label"] = df[" Label"].str.strip()
    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Load CICIDS2017 CSVs → parquet")
    p.add_argument("--raw-dir", default="data/raw", help="Directory with raw CSVs")
    p.add_argument(
        "--out",
        default="data/processed/raw_combined.parquet",
        help="Output parquet path",
    )
    p.add_argument("--config", default="config/model_config.yaml")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    args = _parse_args()
    df = load_cicids2017(raw_dir=args.raw_dir, config_path=args.config)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    logger.info("Saved combined raw data → %s", out)


if __name__ == "__main__":
    main()
