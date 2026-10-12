"""
tests/test_evaluate.py
=======================
Unit tests for src/models/evaluate.py
"""
import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from src.models.evaluate import evaluate_model


@pytest.fixture()
def trained_binary_model():
    """A trivially trained binary classifier."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 10))
    y = (X[:, 0] > 0).astype(int)
    model = DecisionTreeClassifier(random_state=0, max_depth=3)
    model.fit(X, y)
    return model, X, y


class TestEvaluateModel:
    def test_returns_required_keys(self, trained_binary_model, tmp_path, monkeypatch):
        model, X, y = trained_binary_model
        # Redirect plots/metrics to tmp dir
        monkeypatch.chdir(tmp_path)
        (tmp_path / "config").mkdir()
        (tmp_path / "reports/plots").mkdir(parents=True)
        (tmp_path / "reports/metrics").mkdir(parents=True)

        metrics = evaluate_model(
            model=model,
            X_test=X,
            y_test=y,
            model_name="test_model",
            class_names=["BENIGN", "ATTACK"],
        )

        assert "recall_attack" in metrics
        assert "precision_attack" in metrics
        assert "f1_attack" in metrics
        assert "roc_auc" in metrics
        assert "confusion_matrix" in metrics

    def test_recall_is_in_valid_range(self, trained_binary_model, tmp_path, monkeypatch):
        model, X, y = trained_binary_model
        monkeypatch.chdir(tmp_path)
        (tmp_path / "reports/plots").mkdir(parents=True)
        (tmp_path / "reports/metrics").mkdir(parents=True)

        metrics = evaluate_model(model, X, y, model_name="rng_model")
        assert 0.0 <= metrics["recall_attack"] <= 1.0
