"""
src/models/train_binary.py
===========================
Train binary classifiers (BENIGN vs. ATTACK) on preprocessed NIDS data.

Models trained:
- Random Forest
- XGBoost
- LightGBM

Each model is evaluated immediately after training and saved to ``models/``.

Usage (CLI)::

    python -m src.models.train_binary \\
        --data-dir data/processed \\
        --config   config/model_config.yaml

Usage (API)::

    from src.models.train_binary import train_binary
    results = train_binary()
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

from src.features.balancer import balance, get_class_weights
from src.models.evaluate import evaluate_model

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
# Model builders
# ---------------------------------------------------------------------------

def _build_random_forest(params: dict, class_weight: dict | str = "balanced"):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=params.get("n_estimators", 200),
        max_depth=params.get("max_depth", 20),
        min_samples_leaf=params.get("min_samples_leaf", 4),
        n_jobs=params.get("n_jobs", -1),
        class_weight=class_weight,
        random_state=params.get("random_state", 42),
    )


def _build_xgboost(params: dict, n_classes: int = 2):
    from xgboost import XGBClassifier
    objective = "binary:logistic" if n_classes == 2 else "multi:softmax"
    return XGBClassifier(
        n_estimators=params.get("n_estimators", 300),
        max_depth=params.get("max_depth", 8),
        learning_rate=params.get("learning_rate", 0.1),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        eval_metric=params.get("eval_metric", "logloss"),
        objective=objective,
        num_class=n_classes if n_classes > 2 else None,
        n_jobs=params.get("n_jobs", -1),
        random_state=params.get("random_state", 42),
        verbosity=0,
    )


def _build_lightgbm(params: dict, class_weight: dict | str = "balanced"):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=params.get("n_estimators", 300),
        max_depth=params.get("max_depth", 8),
        learning_rate=params.get("learning_rate", 0.05),
        num_leaves=params.get("num_leaves", 63),
        class_weight=class_weight,
        n_jobs=params.get("n_jobs", -1),
        random_state=params.get("random_state", 42),
        verbosity=-1,
    )


# ---------------------------------------------------------------------------
# Train function
# ---------------------------------------------------------------------------

def train_binary(
    data_dir: str = "data/processed",
    config_path: str = "config/model_config.yaml",
    *,
    models_to_train: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Train binary (BENIGN vs. ATTACK) classifiers.

    Parameters
    ----------
    data_dir:
        Directory with ``train.parquet`` and ``test.parquet``.
    config_path:
        Path to YAML config.
    models_to_train:
        Subset of ``['random_forest', 'xgboost', 'lightgbm']``.
        Defaults to all three.

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

    models_dir = Path(paths_cfg.get("models_dir", "models"))
    models_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    train_df = pd.read_parquet(Path(data_dir) / "train.parquet")
    test_df = pd.read_parquet(Path(data_dir) / "test.parquet")

    label_cols = {"label_binary", "label_multi"}
    feature_cols = [c for c in train_df.columns if c not in label_cols]

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["label_binary"].values
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["label_binary"].values

    logger.info(
        "Loaded data — X_train: %s  X_test: %s  Attack rate train: %.2f%%",
        X_train.shape, X_test.shape, y_train.mean() * 100,
    )

    # --- Imbalance handling ---
    imbalance_strategy = imbalance_cfg.get("strategy", "class_weight")
    class_weight: dict | str = "balanced"

    if imbalance_strategy != "class_weight":
        X_train, y_train = balance(
            X_train, y_train,
            strategy=imbalance_strategy,
            smote_k_neighbors=imbalance_cfg.get("smote_k_neighbors", 5),
            undersample_ratio=imbalance_cfg.get("undersample_ratio", 0.5),
        )
    else:
        class_weight = get_class_weights(y_train)

    all_results: dict[str, dict] = {}

    for model_name in models_to_train:
        logger.info("=== Training: %s (binary) ===", model_name)
        params = model_cfgs.get(model_name, {})

        # Build model
        if model_name == "random_forest":
            model = _build_random_forest(params, class_weight=class_weight)
        elif model_name == "xgboost":
            model = _build_xgboost(params, n_classes=2)
        elif model_name == "lightgbm":
            model = _build_lightgbm(params, class_weight=class_weight)
        else:
            logger.warning("Unknown model %r, skipping.", model_name)
            continue

        model.fit(X_train, y_train)
        logger.info("Training complete for %s.", model_name)

        # Evaluate
        class_names = cfg.get("data", {}).get("class_names_binary", ["BENIGN", "ATTACK"])
        metrics = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            model_name=f"{model_name}_binary",
            class_names=class_names,
            config_path=config_path,
        )
        all_results[model_name] = metrics

        # Save model
        save_path = models_dir / f"{model_name}_binary.joblib"
        joblib.dump(model, save_path)
        logger.info("Model saved → %s", save_path)

    # Save feature column list
    feat_path = models_dir / "feature_cols.json"
    feat_path.write_text(json.dumps(feature_cols, indent=2))

    # Save summary
    summary_path = Path(eval_cfg.get("metrics_dir", "reports/metrics")) / "binary_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as fh:
        json.dump(all_results, fh, indent=2, default=str)
    logger.info("Binary training summary → %s", summary_path)

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train binary NIDS classifiers")
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
    results = train_binary(
        data_dir=args.data_dir,
        config_path=args.config,
        models_to_train=args.models,
    )
    logger.info("=== Final binary results ===")
    for model_name, metrics in results.items():
        logger.info("%s — F1: %.4f  Recall: %.4f  AUC: %.4f",
                    model_name,
                    metrics.get("f1_attack", float("nan")),
                    metrics.get("recall_attack", float("nan")),
                    metrics.get("roc_auc", float("nan")))


if __name__ == "__main__":
    main()
