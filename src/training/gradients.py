"""Parameter-shift rule para gradientes exactos."""

from __future__ import annotations

import numpy as np

from src.training.loss import binary_cross_entropy, z_to_prob


def _batch_loss(params: np.ndarray, X: np.ndarray, y: np.ndarray, n_layers: int, circuit_fn) -> float:
    z = np.array([circuit_fn(x, params, n_layers) for x in X], dtype=np.float64)
    p = z_to_prob(z)
    return binary_cross_entropy(p, y)


def parameter_shift_grad(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    n_layers: int = 3,
    circuit_fn=None,
    shift: float = float(np.pi / 2),
) -> np.ndarray:
    """Gradiente via parameter-shift con regla de la cadena.

    El shift rule da dz/dθ exactamente: (z(θ+π/2)-z(θ-π/2))/2.
    La pérdida BCE es no lineal en z, así que no se puede aplicar
    shift directo a L(θ). Se usa cadena: dL/dθ = mean(dL/dz * dz/dθ).
    """
    if circuit_fn is None:
        from src.quantum.numpy_backend.state import expectation_z as circuit_fn  # type: ignore

    # forward para obtener z y p actuales (para dL/dz)
    z = np.array([circuit_fn(x, params, n_layers) for x in X], dtype=np.float64)
    p = np.asarray(z_to_prob(z), dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    # dL/dz por sample: dL/dp * dp/dz,  dp/dz=0.5
    eps = 1e-7
    p_clip = np.clip(p, eps, 1 - eps)
    dL_dp = -(y_arr / p_clip - (1 - y_arr) / (1 - p_clip)) / len(y_arr)
    dL_dz = dL_dp * 0.5

    grad = np.zeros_like(params, dtype=np.float64)
    for i in range(len(params)):
        plus = params.copy()
        minus = params.copy()
        plus[i] += shift
        minus[i] -= shift
        z_plus = np.array([circuit_fn(x, plus, n_layers) for x in X], dtype=np.float64)
        z_minus = np.array([circuit_fn(x, minus, n_layers) for x in X], dtype=np.float64)
        dz_dtheta = (z_plus - z_minus) / 2.0
        grad[i] = float(np.dot(dL_dz, dz_dtheta))
    return grad
