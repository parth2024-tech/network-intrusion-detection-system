"""
src/explainability/shap_explainer.py
======================================
SHAP-based model explainability for NIDS.

Provides:
- Global feature importance (beeswarm / bar plots)
- Local explanation for a single prediction
- Export of SHAP values as a DataFrame

Usage (API)::

    from src.explainability.shap_explainer import SHAPExplainer
    exp = SHAPExplainer(model, X_train, feature_names=feature_cols)
    exp.plot_global_importance()
    local_df = exp.explain_single(X_test[42])
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    Wrapper around SHAP TreeExplainer / KernelExplainer for NIDS models.

    Parameters
    ----------
    model:
        Trained sklearn-compatible model.
    X_background:
        Background dataset used by SHAP (training set sample, ≤ 1000 rows).
    feature_names:
        Column names corresponding to X columns.
    max_background_samples:
        Cap on background samples for KernelExplainer.
    """

    def __init__(
        self,
        model: Any,
        X_background: np.ndarray,
        feature_names: list[str],
        *,
        max_background_samples: int = 500,
    ) -> None:
        try:
            import shap
        except ImportError as exc:
            raise ImportError(
                "shap is required. Install with: pip install shap"
            ) from exc

        self.model = model
        self.feature_names = feature_names

        # Sample background to keep SHAP tractable
        if len(X_background) > max_background_samples:
            idx = np.random.choice(
                len(X_background), max_background_samples, replace=False
            )
            X_bg = X_background[idx]
        else:
            X_bg = X_background

        # Choose appropriate explainer type
        if _is_tree_model(model):
            self.explainer = shap.TreeExplainer(model)
            logger.info("Using SHAP TreeExplainer.")
        else:
            self.explainer = shap.KernelExplainer(
                model.predict_proba, shap.kmeans(X_bg, min(50, len(X_bg)))
            )
            logger.info("Using SHAP KernelExplainer with %d background samples.", len(X_bg))

    # ------------------------------------------------------------------
    # Compute SHAP values
    # ------------------------------------------------------------------

    def compute_shap_values(
        self, X: np.ndarray, *, check_additivity: bool = False
    ) -> np.ndarray:
        """
        Compute SHAP values for a batch of samples.

        Parameters
        ----------
        X:
            Input features, shape (n_samples, n_features).
        check_additivity:
            Passed to TreeExplainer (can be slow for large models).

        Returns
        -------
        np.ndarray
            SHAP values array. Shape: (n_samples, n_features) for binary,
            (n_classes, n_samples, n_features) for multiclass TreeExplainer.
        """
        shap_vals = self.explainer.shap_values(X, check_additivity=check_additivity)
        return shap_vals

    # ------------------------------------------------------------------
    # Global importance plot
    # ------------------------------------------------------------------

    def plot_global_importance(
        self,
        X: np.ndarray,
        *,
        max_display: int = 20,
        plot_type: str = "bar",
        save_path: str | Path | None = "reports/plots/shap_global.png",
        class_index: int = 1,
    ) -> None:
        """
        Generate a SHAP global feature importance plot.

        Parameters
        ----------
        X:
            Samples to compute SHAP values for.
        max_display:
            Number of top features to show.
        plot_type:
            ``'bar'`` (mean |SHAP|) or ``'beeswarm'``.
        save_path:
            If provided, save the figure to this path.
        class_index:
            For multiclass, which class index to plot (default 1 = attack).
        """
        import shap

        shap_values = self.compute_shap_values(X)

        # Handle multiclass output (list of arrays)
        if isinstance(shap_values, list):
            sv = shap_values[class_index]
        else:
            sv = shap_values

        expl = shap.Explanation(
            values=sv,
            data=X,
            feature_names=self.feature_names,
        )

        plt.figure()
        if plot_type == "bar":
            shap.plots.bar(expl, max_display=max_display, show=False)
        else:
            shap.plots.beeswarm(expl, max_display=max_display, show=False)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            logger.info("SHAP global plot → %s", save_path)
        plt.close()

    # ------------------------------------------------------------------
    # Local explanation
    # ------------------------------------------------------------------

    def explain_single(
        self,
        x: np.ndarray,
        *,
        class_index: int = 1,
    ) -> pd.DataFrame:
        """
        Return a DataFrame of feature contributions for a single sample.

        Parameters
        ----------
        x:
            Single sample, shape (n_features,).
        class_index:
            For multiclass, which class to explain.

        Returns
        -------
        pd.DataFrame
            Columns: ``feature``, ``value``, ``shap_value``  
            sorted by |shap_value| descending.
        """
        x_2d = x.reshape(1, -1)
        shap_vals = self.compute_shap_values(x_2d)

        if isinstance(shap_vals, list):
            sv_row = shap_vals[class_index][0]
        else:
            sv_row = shap_vals[0] if shap_vals.ndim == 2 else shap_vals

        df = pd.DataFrame({
            "feature": self.feature_names,
            "value": x,
            "shap_value": sv_row,
        }).assign(abs_shap=lambda d: d["shap_value"].abs())
        return df.sort_values("abs_shap", ascending=False).drop(columns="abs_shap")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_tree_model(model: Any) -> bool:
    """Return True if model is a tree-based estimator compatible with TreeExplainer."""
    tree_types = (
        "RandomForestClassifier",
        "RandomForestRegressor",
        "XGBClassifier",
        "XGBRegressor",
        "LGBMClassifier",
        "LGBMRegressor",
        "GradientBoostingClassifier",
        "ExtraTreesClassifier",
        "DecisionTreeClassifier",
    )
    return type(model).__name__ in tree_types
