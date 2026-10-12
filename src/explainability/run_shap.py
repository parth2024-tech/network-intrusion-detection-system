"""
src/explainability/run_shap.py
================================
CLI script to generate SHAP global importance plots
for trained NIDS models and save to reports/plots/.

Usage::

    python -m src.explainability.run_shap \\
        --model   random_forest \\
        --mode    binary \\
        --n-samples 1000
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.explainability.shap_explainer import SHAPExplainer

logger = logging.getLogger(__name__)


def run_shap(
    model_name: str = "random_forest",
    mode: str = "binary",
    data_dir: str = "data/processed",
    models_dir: str = "models",
    n_samples: int = 1000,
    plots_dir: str = "reports/plots",
) -> None:
    """
    Generate and save SHAP plots for a trained model.

    Parameters
    ----------
    model_name:
        One of ``random_forest``, ``xgboost``, ``lightgbm``.
    mode:
        ``binary`` or ``multiclass``.
    data_dir:
        Directory with ``train.parquet`` and ``test.parquet``.
    models_dir:
        Directory containing saved ``.joblib`` models.
    n_samples:
        Number of test samples to compute SHAP values on.
    plots_dir:
        Output directory for SHAP plots.
    """
    model_path = Path(models_dir) / f"{model_name}_{mode}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)
    logger.info("Loaded model: %s", model_path)

    # Load data
    train_df = pd.read_parquet(Path(data_dir) / "train.parquet")
    test_df = pd.read_parquet(Path(data_dir) / "test.parquet")

    feat_path = Path(models_dir) / "feature_cols.json"
    if feat_path.exists():
        feature_cols = json.loads(feat_path.read_text())
    else:
        label_cols = {"label_binary", "label_multi"}
        feature_cols = [c for c in train_df.columns if c not in label_cols]

    X_train = train_df[[c for c in feature_cols if c in train_df.columns]].values.astype(np.float32)
    X_test = test_df[[c for c in feature_cols if c in test_df.columns]].values.astype(np.float32)

    # Sample test set
    if len(X_test) > n_samples:
        idx = np.random.choice(len(X_test), n_samples, replace=False)
        X_test = X_test[idx]

    explainer = SHAPExplainer(
        model=model,
        X_background=X_train,
        feature_names=feature_cols,
        max_background_samples=500,
    )

    out_dir = Path(plots_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    explainer.plot_global_importance(
        X_test,
        plot_type="bar",
        save_path=out_dir / f"shap_{model_name}_{mode}_bar.png",
    )
    explainer.plot_global_importance(
        X_test,
        plot_type="beeswarm",
        save_path=out_dir / f"shap_{model_name}_{mode}_beeswarm.png",
    )
    logger.info("SHAP plots saved to %s", out_dir)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate SHAP explanations for NIDS models")
    p.add_argument("--model", default="random_forest",
                   choices=["random_forest", "xgboost", "lightgbm"])
    p.add_argument("--mode", default="binary", choices=["binary", "multiclass"])
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--plots-dir", default="reports/plots")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    args = _parse_args()
    run_shap(
        model_name=args.model,
        mode=args.mode,
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        n_samples=args.n_samples,
        plots_dir=args.plots_dir,
    )


if __name__ == "__main__":
    main()
