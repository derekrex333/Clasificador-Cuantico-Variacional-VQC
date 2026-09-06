"""Normalización a [0, π] y split estratificado."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def normalize_to_pi(
    X: np.ndarray,
    feature_range: tuple[float, float] = (0.0, float(np.pi)),
) -> tuple[np.ndarray, MinMaxScaler]:
    """Escala cada feature a [0, π] (rango para Ry angle encoding).

    Returns
    -------
    X_scaled, scaler (por si se necesita transform inverso o aplicar a test separado)
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    X_scaled = scaler.fit_transform(X)
    return X_scaled.astype(np.float64), scaler


def apply_scaler(X: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    return scaler.transform(X).astype(np.float64)


def train_test_split_stratified(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
