import pytest
import numpy as np
from genre_classifier.model import MFCCDataModule


# Test return types
def test_prepare_datasets_returns():
    X = np.random.rand(100, 20)
    y = np.random.randint(0, 5, size=100)

    X_train, y_train, X_test, y_test, X_val, y_val = MFCCDataModule.prepare_datasets(
        X, y, test_size=0.2, validation_size=0.2
    )

    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(X_val, np.ndarray)
    assert isinstance(y_val, np.ndarray)


# Test output shapes
@pytest.mark.parametrize(
    "n_samples, n_train, n_test, n_val",
    [(100, 80, 10, 10), (200, 160, 20, 20), (500, 400, 50, 50)],
)
def test_prepare_datasets_shapes(n_samples, n_train, n_test, n_val):
    n_features = 20
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(0, 5, size=n_samples)

    X_train, y_train, X_test, y_test, X_val, y_val = MFCCDataModule.prepare_datasets(
        X, y, test_size=n_test, validation_size=n_val
    )

    assert X_train.shape == (n_train, n_features)
    assert y_train.shape == (n_train,)
    assert X_test.shape == (n_test, n_features)
    assert y_test.shape == (n_test,)
    assert X_val.shape == (n_val, n_features)
    assert y_val.shape == (n_val,)


# Test shuffling
def test_prepare_datasets_shuffle():
    # Create dummy data
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 2, 3])

    n_test, n_val = 0.1, 0.1

    # Shuffle with seed
    np.random.seed(42)
    X_train, y_train, X_test, y_test, X_val, y_val = MFCCDataModule.prepare_datasets(
        X, y, shuffle=True, test_size=n_test, validation_size=n_val
    )
    shuffled_order = [y_train[0], y_val[0]]

    # No shuffling
    X_train, y_train, X_test, y_test, X_val, y_val = MFCCDataModule.prepare_datasets(
        X, y, shuffle=False, test_size=n_test, validation_size=n_val
    )
    no_shuffle_order = [y_train[0], y_val[0]]

    assert shuffled_order != no_shuffle_order
