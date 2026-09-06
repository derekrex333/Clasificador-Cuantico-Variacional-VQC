"""Mapeo <Z>→probabilidad y BCE."""

from __future__ import annotations

import numpy as np


def z_to_prob(z: np.ndarray | float, eps: float = 1e-7) -> np.ndarray | float:
    p = (np.asarray(z) + 1.0) / 2.0
    return np.clip(p, eps, 1 - eps)


def binary_cross_entropy(p: np.ndarray, y: np.ndarray, eps: float = 1e-7) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    y = np.asarray(y, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def accuracy_from_probs(p: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> float:
    y_pred = (np.asarray(p) >= threshold).astype(int)
    return float(np.mean(y_pred == np.asarray(y)))
