"""
tests/test_balancer.py
=======================
Unit tests for src/features/balancer.py
"""
import numpy as np
import pytest

from src.features.balancer import balance, get_class_weights


@pytest.fixture()
def imbalanced_data():
    """1000 majority class, 50 minority class."""
    rng = np.random.default_rng(42)
    X_maj = rng.standard_normal((1000, 10))
    X_min = rng.standard_normal((50, 10))
    X = np.vstack([X_maj, X_min])
    y = np.array([0] * 1000 + [1] * 50)
    return X, y


class TestBalance:
    def test_class_weight_returns_unchanged(self, imbalanced_data):
        X, y = imbalanced_data
        X_res, y_res = balance(X, y, strategy="class_weight")
        assert X_res.shape == X.shape
        assert (y_res == y).all()

    def test_smote_increases_minority(self, imbalanced_data):
        pytest.importorskip("imblearn")
        X, y = imbalanced_data
        X_res, y_res = balance(X, y, strategy="smote")
        assert len(X_res) > len(X)
        minority_count = (y_res == 1).sum()
        assert minority_count > 50

    def test_unknown_strategy_raises(self, imbalanced_data):
        X, y = imbalanced_data
        with pytest.raises(ValueError, match="Unknown strategy"):
            balance(X, y, strategy="magic")


class TestGetClassWeights:
    def test_minority_has_higher_weight(self, imbalanced_data):
        _, y = imbalanced_data
        weights = get_class_weights(y)
        # minority class (1) should weigh more
        assert weights[1] > weights[0]

    def test_returns_dict_with_both_classes(self, imbalanced_data):
        _, y = imbalanced_data
        weights = get_class_weights(y)
        assert 0 in weights
        assert 1 in weights
