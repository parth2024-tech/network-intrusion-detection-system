"""
src/models/train_multiclass.py
================================
Train multiclass classifiers — identify *which* attack type a flow belongs to.

Classes (per model_config.yaml):
  0 = BENIGN  1 = Brute Force  2 = DoS/DDoS  3 = Heartbleed
  4 = Web Attack  5 = Infiltration  6 = Botnet  7 = Port Scan

Usage (CLI)::

    python -m src.models.train_multiclass \\
        --data-dir data/processed \\
        --config   config/model_config.yaml

Usage (API)::

    from src.models.train_multiclass import train_multiclass
    results = train_multiclass()
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from src.features.balancer import balance
from src.models.evaluate import compare_models, evaluate_model

logger = logging.getLogger(__name__)


def _load_config(config_path: str) -> dict:
    p = Path(config_path)
    if not p.exists():
        return {}
    with p.open() as fh:
        return yaml.safe_load(fh)


def train_multiclass(
    data_dir: str = "data/processed",
    config_path: str = "config/model_config.yaml",
    *,
    models_to_train: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Train multiclass classifiers (attack type identification).

    Parameters
    ----------
    data_dir, config_path, models_to_train:
        See :func:`src.models.train_binary.train_binary`.

    Returns
    -------
    dict
        ``{model_name: evaluation_metrics_dict}``
    """
    if models_to_train is None:
        models_to_train = ["random_forest", "xgboost", "lightgbm"]

    cfg = _load_config(config_path)
    model_cfgs = cfg.get("models", {})
    imbalance_cfg = cfg.get("imbalance", {})
    paths_cfg = cfg.get("paths", {})
    eval_cfg = cfg.get("evaluation", {})
    class_names: list[str] = cfg.get("data", {}).get(
        "class_names_multi",
        ["BENIGN", "Brute Force", "DoS/DDoS", "Heartbleed",
         "Web Attack", "Infiltration", "Botnet", "Port Scan"]
    )

    models_dir = Path(paths_cfg.get("models_dir", "models"))
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_df = pd.read_parquet(Path(data_dir) / "train.parquet")
    test_df = pd.read_parquet(Path(data_dir) / "test.parquet")

    label_cols = {"label_binary", "label_multi"}
    feature_cols = [c for c in train_df.columns if c not in label_cols]

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["label_multi"].values
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["label_multi"].values

    # Remove rows with unknown label (-1) from train
    valid_mask = y_train != -1
    X_train, y_train = X_train[valid_mask], y_train[valid_mask]
    valid_mask_t = y_test != -1
    X_test, y_test = X_test[valid_mask_t], y_test[valid_mask_t]

    logger.info(
        "Multiclass — X_train: %s  classes: %s",
        X_train.shape, np.unique(y_train),
    )

    # Imbalance handling
    strategy = imbalance_cfg.get("strategy", "class_weight")
    class_weight: str | dict = "balanced"
    if strategy != "class_weight":
        X_train, y_train = balance(
            X_train, y_train,
            strategy=strategy,
            smote_k_neighbors=imbalance_cfg.get("smote_k_neighbors", 5),
        )

    n_classes = len(np.unique(y_train))
    all_results: dict[str, dict] = {}

    for model_name in models_to_train:
        logger.info("=== Training: %s (multiclass, %d classes) ===", model_name, n_classes)
        params = model_cfgs.get(model_name, {})

        if model_name == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=params.get("n_estimators", 200),
                max_depth=params.get("max_depth", 20),
                min_samples_leaf=params.get("min_samples_leaf", 4),
                n_jobs=params.get("n_jobs", -1),
                class_weight=class_weight,
                random_state=params.get("random_state", 42),
            )
        elif model_name == "xgboost":
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=params.get("n_estimators", 300),
                max_depth=params.get("max_depth", 8),
                learning_rate=params.get("learning_rate", 0.1),
                subsample=params.get("subsample", 0.8),
                colsample_bytree=params.get("colsample_bytree", 0.8),
                objective="multi:softmax",
                num_class=n_classes,
                eval_metric="mlogloss",
                n_jobs=params.get("n_jobs", -1),
                random_state=params.get("random_state", 42),
                verbosity=0,
            )
        elif model_name == "lightgbm":
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(
                n_estimators=params.get("n_estimators", 300),
                max_depth=params.get("max_depth", 8),
                learning_rate=params.get("learning_rate", 0.05),
                num_leaves=params.get("num_leaves", 63),
                class_weight=class_weight,
                n_jobs=params.get("n_jobs", -1),
                random_state=params.get("random_state", 42),
                verbosity=-1,
            )
        else:
            logger.warning("Unknown model %r, skipping.", model_name)
            continue

        model.fit(X_train, y_train)

        # Only pass class names for the classes that exist in y_test
        present_classes = sorted(np.unique(y_test).tolist())
        names_present = [
            class_names[i] if i < len(class_names) else str(i)
            for i in present_classes
        ]

        metrics = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            model_name=f"{model_name}_multiclass",
            class_names=names_present,
            config_path=config_path,
        )
        all_results[model_name] = metrics

        save_path = models_dir / f"{model_name}_multiclass.joblib"
        joblib.dump(model, save_path)
        logger.info("Model saved → %s", save_path)

    # Comparison chart
    compare_models(
        all_results,
        save_path=str(
            Path(eval_cfg.get("plots_dir", "reports/plots")) / "multiclass_comparison.png"
        ),
    )

    summary_path = Path(eval_cfg.get("metrics_dir", "reports/metrics")) / "multiclass_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as fh:
        json.dump(all_results, fh, indent=2, default=str)

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train multiclass NIDS classifiers")
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--config", default="config/model_config.yaml")
    p.add_argument(
        "--models",
        nargs="+",
        choices=["random_forest", "xgboost", "lightgbm"],
        default=["random_forest", "xgboost", "lightgbm"],
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()
    results = train_multiclass(
        data_dir=args.data_dir,
        config_path=args.config,
        models_to_train=args.models,
    )
    logger.info("=== Multiclass training complete ===")
    for name, m in results.items():
        logger.info("%s — Macro F1: %.4f", name, m.get("macro_f1", float("nan")))


if __name__ == "__main__":
    main()
