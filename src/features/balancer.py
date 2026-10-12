"""
src/features/balancer.py
=========================
Handle severe class imbalance in the NIDS training set.

Strategies:
- ``class_weight``  — pass ``class_weight='balanced'`` to sklearn models (no resampling)
- ``smote``         — Synthetic Minority Over-sampling Technique
- ``undersample``   — Random under-sampling of the majority class
- ``combined``      — SMOTE + Tomek Links

Usage (API)::

    from src.features.balancer import balance
    X_res, y_res = balance(X_train, y_train, strategy="smote")
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def balance(
    X: np.ndarray,
    y: np.ndarray,
    strategy: str = "class_weight",
    smote_k_neighbors: int = 5,
    undersample_ratio: float = 0.5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample the training set to mitigate class imbalance.

    .. warning::
       **Only call on training data.** Never resample the test set.

    Parameters
    ----------
    X:
        Training feature matrix.
    y:
        Training labels (binary or multiclass integer array).
    strategy:
        One of:

        * ``'class_weight'`` — no resampling; returns unchanged *X*, *y*.
          Pass ``class_weight='balanced'`` to your sklearn model instead.
        * ``'smote'`` — oversample minority classes.
        * ``'undersample'`` — random undersample majority class.
        * ``'combined'`` — SMOTE + Tomek Links clean-up.
    smote_k_neighbors:
        Number of nearest neighbours used by SMOTE.
    undersample_ratio:
        Ratio of minority to majority class size after undersampling.
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    (X_resampled, y_resampled)
    """
    _log_distribution("Before balancing", y)

    if strategy == "class_weight":
        logger.info(
            "Strategy=class_weight: no resampling. "
            "Pass class_weight='balanced' to your model."
        )
        return X, y

    try:
        from imblearn.combine import SMOTETomek
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
    except ImportError as exc:
        raise ImportError(
            "imbalanced-learn is required for SMOTE/undersample strategies. "
            "Run: pip install imbalanced-learn"
        ) from exc

    if strategy == "smote":
        sampler = SMOTE(k_neighbors=smote_k_neighbors, random_state=random_state)
    elif strategy == "undersample":
        sampler = RandomUnderSampler(
            sampling_strategy=undersample_ratio, random_state=random_state
        )
    elif strategy == "combined":
        sampler = SMOTETomek(
            smote=SMOTE(k_neighbors=smote_k_neighbors, random_state=random_state),
            random_state=random_state,
        )
    else:
        raise ValueError(
            f"Unknown strategy: {strategy!r}. "
            "Choose from: class_weight, smote, undersample, combined"
        )

    X_res, y_res = sampler.fit_resample(X, y)
    _log_distribution("After balancing", y_res)
    return X_res, y_res


def get_class_weights(y: np.ndarray) -> dict[int, float]:
    """
    Compute balanced class weights as a dict suitable for sklearn's
    ``class_weight`` parameter.

    Parameters
    ----------
    y:
        Label array.

    Returns
    -------
    dict mapping class index → weight
    """
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    cw = dict(zip(classes.tolist(), weights.tolist()))
    logger.info("Class weights: %s", cw)
    return cw


def _log_distribution(label: str, y: np.ndarray) -> None:
    unique, counts = np.unique(y, return_counts=True)
    info = ", ".join(f"class {c}: {n}" for c, n in zip(unique, counts))
    logger.info("%s — %s", label, info)
