"""
src/anomaly/autoencoder.py
===========================
Autoencoder-based anomaly detection for NIDS.

The autoencoder is trained **only on benign traffic** (class 0).
At inference time, high reconstruction error signals a potential anomaly
(zero-day / unknown attack pattern).

Uses scikit-learn compatible interface for integration with the pipeline.

Usage (API)::

    from src.anomaly.autoencoder import NIDSAutoencoder
    ae = NIDSAutoencoder(hidden_dims=[64, 32, 16, 32, 64])
    ae.fit(X_benign)
    scores = ae.reconstruction_error(X_test)
    y_pred = ae.predict(X_test, threshold=ae.threshold_)
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class NIDSAutoencoder:
    """
    Shallow MLP Autoencoder for anomaly detection in network flows.

    Trained exclusively on benign traffic. Flows with reconstruction
    mean-squared-error above ``threshold_`` are flagged as anomalous.

    Parameters
    ----------
    hidden_dims:
        Sequence of layer sizes for encoder and decoder
        (symmetric: e.g. [64, 32, 16, 32, 64]).
    epochs:
        Number of training epochs.
    batch_size:
        Mini-batch size.
    learning_rate:
        Adam learning rate.
    threshold_percentile:
        Percentile of training reconstruction errors used as the
        anomaly decision threshold.
    random_state:
        Random seed.
    """

    def __init__(
        self,
        hidden_dims: Sequence[int] = (64, 32, 16, 32, 64),
        epochs: int = 50,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        threshold_percentile: float = 95.0,
        random_state: int = 42,
    ) -> None:
        self.hidden_dims = list(hidden_dims)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.threshold_percentile = threshold_percentile
        self.random_state = random_state

        self.model_: Any = None
        self.threshold_: float | None = None
        self.input_dim_: int | None = None

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_model(self, input_dim: int):
        """Build a Keras Sequential autoencoder."""
        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is required for the Autoencoder. "
                "Install it with: pip install tensorflow"
            ) from exc

        tf.random.set_seed(self.random_state)
        model = keras.Sequential(name="NIDS_Autoencoder")
        model.add(keras.layers.Input(shape=(input_dim,)))
        for units in self.hidden_dims:
            model.add(keras.layers.Dense(units, activation="relu"))
        model.add(keras.layers.Dense(input_dim, activation="linear"))

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
        )
        model.summary(print_fn=logger.info)
        return model

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(self, X_benign: np.ndarray) -> NIDSAutoencoder:
        """
        Train the autoencoder on benign flows only.

        Parameters
        ----------
        X_benign:
            Feature matrix of **benign** training samples only.

        Returns
        -------
        self
        """
        self.input_dim_ = X_benign.shape[1]
        self.model_ = self._build_model(self.input_dim_)

        logger.info(
            "Training autoencoder on %d benign samples, "
            "input_dim=%d, epochs=%d, batch_size=%d",
            len(X_benign), self.input_dim_, self.epochs, self.batch_size,
        )

        self.model_.fit(
            X_benign, X_benign,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            shuffle=True,
            verbose=1,
        )

        # Set threshold from training reconstruction errors
        errors = self._mse(X_benign)
        self.threshold_ = float(np.percentile(errors, self.threshold_percentile))
        logger.info(
            "Anomaly threshold (p%.0f of training MSE): %.6f",
            self.threshold_percentile, self.threshold_,
        )
        return self

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """
        Compute per-sample mean-squared reconstruction error.

        Parameters
        ----------
        X:
            Feature matrix.

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        if self.model_ is None:
            raise RuntimeError("Call fit() before reconstruction_error().")
        X_reconstructed = self.model_.predict(X, verbose=0)
        return np.mean((X - X_reconstructed) ** 2, axis=1)

    def predict(
        self, X: np.ndarray, threshold: float | None = None
    ) -> np.ndarray:
        """
        Predict anomaly labels.

        Parameters
        ----------
        X:
            Feature matrix.
        threshold:
            Override the fitted threshold. Defaults to ``self.threshold_``.

        Returns
        -------
        np.ndarray
            Binary array: 1 = anomaly, 0 = normal.
        """
        thr = threshold if threshold is not None else self.threshold_
        if thr is None:
            raise RuntimeError("No threshold set. Call fit() first.")
        errors = self.reconstruction_error(X)
        return (errors > thr).astype(int)

    def _mse(self, X: np.ndarray) -> np.ndarray:
        X_hat = self.model_.predict(X, verbose=0)
        return np.mean((X - X_hat) ** 2, axis=1)

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str = "models/autoencoder.keras") -> None:
        """
        Save the Keras model and threshold to disk.

        Parameters
        ----------
        path:
            Path for the ``.keras`` model file. The threshold is saved
            alongside as ``<path>.threshold.npy``.
        """
        if self.model_ is None:
            raise RuntimeError("No model to save. Call fit() first.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model_.save(path)
        np.save(f"{path}.threshold.npy", np.array([self.threshold_]))
        logger.info("Autoencoder saved → %s  (threshold=%.6f)", path, self.threshold_)

    def load(self, path: str = "models/autoencoder.keras") -> NIDSAutoencoder:
        """
        Load a previously saved autoencoder from disk.

        Parameters
        ----------
        path:
            Path to the ``.keras`` model file.

        Returns
        -------
        self
        """
        try:
            from tensorflow import keras
        except ImportError as exc:
            raise ImportError("TensorFlow required to load autoencoder.") from exc

        self.model_ = keras.models.load_model(path)
        threshold_file = f"{path}.threshold.npy"
        if Path(threshold_file).exists():
            self.threshold_ = float(np.load(threshold_file)[0])
        logger.info(
            "Autoencoder loaded ← %s  (threshold=%.6f)", path, self.threshold_
        )
        return self
