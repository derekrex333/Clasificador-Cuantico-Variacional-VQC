"""Loop híbrido — circuito como capa, optimizador clásico por fuera."""

from __future__ import annotations

import numpy as np

from src.quantum.numpy_backend.state import expectation_z
from src.training.gradients import parameter_shift_grad
from src.training.loss import accuracy_from_probs, binary_cross_entropy, z_to_prob


def _forward(X: np.ndarray, params: np.ndarray, n_layers: int):
    z = np.array([expectation_z(x, params, n_layers) for x in X], dtype=np.float64)
    p = z_to_prob(z)
    return z, p


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    n_layers: int = 3,
    lr: float = 0.05,
    epochs: int = 80,
    optimizer: str = "adam",
    batch_size: int | None = None,
    seed: int = 42,
    verbose: bool = False,
) -> dict:
    from src.quantum.numpy_backend.state import init_params, num_params

    n_params = num_params(n_qubits=2, n_layers=n_layers)
    params = init_params(n_qubits=2, n_layers=n_layers, seed=seed)

    # Adam state
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    history = {"loss": [], "acc": [], "val_loss": [], "val_acc": [], "params": []}
    rng = np.random.default_rng(seed)

    for epoch in range(1, epochs + 1):
        # mini-batch o full batch
        if batch_size is not None and batch_size < len(X_train):
            idx = rng.choice(len(X_train), size=batch_size, replace=False)
            Xb, yb = X_train[idx], y_train[idx]
        else:
            Xb, yb = X_train, y_train

        _, p = _forward(Xb, params, n_layers)
        loss = binary_cross_entropy(p, yb)
        acc = accuracy_from_probs(p, yb)

        grad = parameter_shift_grad(params, Xb, yb, n_layers, circuit_fn=expectation_z)

        if optimizer == "adam":
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad**2)
            m_hat = m / (1 - beta1**epoch)
            v_hat = v / (1 - beta2**epoch)
            params = params - lr * m_hat / (np.sqrt(v_hat) + eps)
        elif optimizer == "sgd":
            params = params - lr * grad
        else:
            raise ValueError(f"optimizer desconocido: {optimizer!r}")

        history["loss"].append(loss)
        history["acc"].append(acc)
        history["params"].append(params.copy())

        if X_val is not None and y_val is not None:
            _, p_val = _forward(X_val, params, n_layers)
            history["val_loss"].append(binary_cross_entropy(p_val, y_val))
            history["val_acc"].append(accuracy_from_probs(p_val, y_val))

        if verbose and epoch % 10 == 0:
            print(f"epoch {epoch:3d} loss={loss:.4f} acc={acc:.3f}")

    history["params_final"] = params
    return history


def predict_proba(X: np.ndarray, params: np.ndarray, n_layers: int = 3) -> np.ndarray:
    _, p = _forward(X, params, n_layers)
    return p


def predict(X: np.ndarray, params: np.ndarray, n_layers: int = 3, threshold: float = 0.5) -> np.ndarray:
    return (predict_proba(X, params, n_layers) >= threshold).astype(int)
