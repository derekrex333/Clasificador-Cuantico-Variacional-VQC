import numpy as np
from src.data.preprocessing import normalize_to_pi
from src.data.datasets import get_dataset

def test_normalize_range():
    X, _ = get_dataset("moons", n_samples=20, random_state=0)
    Xs, _ = normalize_to_pi(X)
    assert Xs.min() >= -1e-9
    assert Xs.max() <= np.pi + 1e-9

def test_normalize_shape():
    X = np.array([[0, 1], [2, 3]], dtype=float)
    Xs, scaler = normalize_to_pi(X)
    assert Xs.shape == X.shape
