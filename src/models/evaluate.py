"""
src/models/evaluate.py
=======================
Evaluate a trained sklearn/XGBoost model on the test set.

Computes:
- Classification report (precision, recall, F1 per class)
- ROC-AUC (binary) or macro ROC-AUC (multiclass)
- Confusion matrix (saved as a heatmap PNG)

Usage (API)::

    from src.models.evaluate import evaluate_model
    metrics = evaluate_model(model, X_test, y_test, model_name="rf_binary")
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

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
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    model_name: str = "model",
    class_names: list[str] | None = None,
    config_path: str = "config/model_config.yaml",
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Evaluate a classification model and save plots.

    Parameters
    ----------
    model:
        Trained sklearn-compatible model with ``predict`` and
        (optionally) ``predict_proba``.
    X_test:
        Test feature matrix.
    y_test:
        True labels.
    model_name:
        Used for file naming in ``reports/``.
    class_names:
        Human-readable class names for confusion matrix axes.
    config_path:
        YAML config path.
    threshold:
        Decision threshold for binary classifiers (applied to proba[:,1]).

    Returns
    -------
    dict
        Keys: ``precision_attack``, ``recall_attack``, ``f1_attack``,
        ``roc_auc``, ``report``, ``confusion_matrix``.
    """
    cfg = _load_config(config_path)
    eval_cfg = cfg.get("evaluation", {})
    plots_dir = Path(eval_cfg.get("plots_dir", "reports/plots"))
    metrics_dir = Path(eval_cfg.get("metrics_dir", "reports/metrics"))
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    is_binary = len(np.unique(y_test)) <= 2

    # Predict
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        if is_binary:
            y_pred = (y_proba[:, 1] >= threshold).astype(int)
        else:
            y_pred = np.argmax(y_proba, axis=1)
    else:
        y_pred = model.predict(X_test)
        y_proba = None

    # Classification report
    report = classification_report(
        y_test, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    logger.info("\n%s", classification_report(
        y_test, y_pred, target_names=class_names, zero_division=0
    ))

    # ROC-AUC
    roc_auc = float("nan")
    if y_proba is not None:
        try:
            if is_binary:
                roc_auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                roc_auc = roc_auc_score(
                    y_test, y_proba, multi_class="ovr", average="macro"
                )
            logger.info("ROC-AUC (%s): %.4f", model_name, roc_auc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ROC-AUC failed: %s", exc)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    _plot_confusion_matrix(
        cm, class_names=class_names,
        title=f"Confusion Matrix — {model_name}",
        save_path=plots_dir / f"{model_name}_cm.png",
    )

    # Compact metrics dict
    attack_key = class_names[1] if class_names and len(class_names) > 1 else "1"
    attack_metrics = report.get(attack_key, report.get("1", {}))

    metrics = {
        "model": model_name,
        "precision_attack": attack_metrics.get("precision", float("nan")),
        "recall_attack": attack_metrics.get("recall", float("nan")),
        "f1_attack": attack_metrics.get("f1-score", float("nan")),
        "roc_auc": roc_auc,
        "accuracy": report.get("accuracy", float("nan")),
        "macro_f1": report.get("macro avg", {}).get("f1-score", float("nan")),
        "report": report,
        "confusion_matrix": cm.tolist(),
    }

    # Save metrics JSON
    with (metrics_dir / f"{model_name}_metrics.json").open("w") as fh:
        json.dump(metrics, fh, indent=2, default=str)

    return metrics


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str] | None,
    title: str,
    save_path: Path,
) -> None:
    """Save a seaborn-styled confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(max(6, len(cm) * 1.5), max(5, len(cm) * 1.2)))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names or "auto",
        yticklabels=class_names or "auto",
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved → %s", save_path)


def plot_roc_curve(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "model",
    plots_dir: str = "reports/plots",
) -> None:
    """
    Plot and save a ROC curve (binary only).

    Parameters
    ----------
    model:
        Trained binary classifier with ``predict_proba``.
    X_test, y_test:
        Test data.
    model_name:
        Used for the filename.
    plots_dir:
        Directory to save the plot.
    """
    from sklearn.metrics import RocCurveDisplay

    plots_path = Path(plots_dir)
    plots_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=model_name)
    ax.set_title(f"ROC Curve — {model_name}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.tight_layout()
    save_path = plots_path / f"{model_name}_roc.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("ROC curve saved → %s", save_path)


def compare_models(
    results: dict[str, dict],
    save_path: str = "reports/plots/model_comparison.png",
) -> None:
    """
    Bar-chart comparing recall, precision, F1, and AUC across models.

    Parameters
    ----------
    results:
        ``{model_name: metrics_dict}`` from :func:`evaluate_model`.
    save_path:
        Output PNG path.
    """
    import pandas as pd

    rows = []
    for name, m in results.items():
        rows.append({
            "Model": name,
            "Recall (Attack)": m.get("recall_attack", 0),
            "Precision (Attack)": m.get("precision_attack", 0),
            "F1 (Attack)": m.get("f1_attack", 0),
            "ROC-AUC": m.get("roc_auc", 0),
        })

    df = pd.DataFrame(rows).set_index("Model")
    fig, ax = plt.subplots(figsize=(10, 5))
    df.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="black", linewidth=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Binary Classification")
    ax.legend(loc="lower right")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Model comparison chart → %s", save_path)
