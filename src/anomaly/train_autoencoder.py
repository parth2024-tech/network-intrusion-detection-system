"""
src/anomaly/train_autoencoder.py
==================================
CLI / API entry point to train the NIDS autoencoder on benign flows.

Usage (CLI)::

    python -m src.anomaly.train_autoencoder \\
        --data-dir data/processed \\
        --config   config/model_config.yaml

Usage (API)::

    from src.anomaly.train_autoencoder import train_autoencoder
    ae = train_autoencoder()
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.anomaly.autoencoder import NIDSAutoencoder

logger = logging.getLogger(__name__)


def _load_config(config_path: str) -> dict:
    p = Path(config_path)
    if not p.exists():
        return {}
    with p.open() as fh:
        return yaml.safe_load(fh)


def train_autoencoder(
    data_dir: str = "data/processed",
    config_path: str = "config/model_config.yaml",
) -> NIDSAutoencoder:
    """
    Train the autoencoder on benign-only training flows and save to disk.

    Parameters
    ----------
    data_dir:
        Directory containing ``train.parquet``.
    config_path:
        YAML config path.

    Returns
    -------
    NIDSAutoencoder
        Fitted autoencoder.
    """
    cfg = _load_config(config_path)
    ae_cfg = cfg.get("autoencoder", {})
    paths_cfg = cfg.get("paths", {})

    # Load benign training data only
    train_df = pd.read_parquet(Path(data_dir) / "train.parquet")
    label_cols = {"label_binary", "label_multi"}
    feature_cols = [c for c in train_df.columns if c not in label_cols]

    benign_mask = train_df["label_binary"] == 0
    X_benign = train_df.loc[benign_mask, feature_cols].values.astype(np.float32)
    logger.info("Benign training samples: %d  features: %d", *X_benign.shape)

    ae = NIDSAutoencoder(
        hidden_dims=ae_cfg.get("hidden_dims", [64, 32, 16, 32, 64]),
        epochs=ae_cfg.get("epochs", 50),
        batch_size=ae_cfg.get("batch_size", 256),
        learning_rate=ae_cfg.get("learning_rate", 1e-3),
        threshold_percentile=ae_cfg.get("reconstruction_threshold_percentile", 95),
        random_state=ae_cfg.get("random_state", 42),
    )
    ae.fit(X_benign)

    save_path = paths_cfg.get("autoencoder", "models/autoencoder.keras")
    ae.save(save_path)

    return ae


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train NIDS autoencoder")
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--config", default="config/model_config.yaml")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()
    train_autoencoder(data_dir=args.data_dir, config_path=args.config)


if __name__ == "__main__":
    main()
