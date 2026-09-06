"""Simulador de estado para 2 qubits — encoding + ansatz hardware-efficient."""

from __future__ import annotations

import numpy as np

from src.quantum.numpy_backend.gates import cnot, ry, rz, single_qubit_gate_on_state

N_QUBITS = 2


def num_params(n_qubits: int = 2, n_layers: int = 3) -> int:
    """2 * n * L parámetros (Ry+Rz por qubit por capa)."""
    return 2 * n_qubits * n_layers


def init_params(n_qubits: int = 2, n_layers: int = 3, seed: int = 42, scale: float = 0.1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-scale * np.pi, scale * np.pi, size=num_params(n_qubits, n_layers)).astype(np.float64)


def _apply_ansatz(state: np.ndarray, params: np.ndarray, n_layers: int) -> np.ndarray:
    """Aplica L capas: Ry(θ)Rz(φ) por qubit + CNOT(0→1)."""
    n_qubits = N_QUBITS
    idx = 0
    for _ in range(n_layers):
        for q in range(n_qubits):
            theta = params[idx]
            phi = params[idx + 1]
            idx += 2
            # Ry luego Rz (orden: Rz @ Ry |ψ> equivale a aplicar Ry primero)
            U = rz(phi) @ ry(theta)
            full = single_qubit_gate_on_state(U, q, n_qubits)
            state = full @ state
        # entrelazamiento
        state = cnot() @ state
    return state


def state_vector(x: np.ndarray, params: np.ndarray, n_layers: int = 3) -> np.ndarray:
    """Vector de estado |ψ(x,θ)> para 2 features."""
    assert x.shape[0] == N_QUBITS, f"esperaba {N_QUBITS} features, got {x.shape}"
    assert params.shape[0] == num_params(N_QUBITS, n_layers)
    # |00>
    state = np.zeros(2**N_QUBITS, dtype=np.complex128)
    state[0] = 1.0
    # angle encoding: Ry(x_i) por qubit
    for q in range(N_QUBITS):
        full = single_qubit_gate_on_state(ry(float(x[q])), q, N_QUBITS)
        state = full @ state
    # ansatz
    state = _apply_ansatz(state, params, n_layers)
    return state


def expectation_z(x: np.ndarray, params: np.ndarray, n_layers: int = 3) -> float:
    """<Z0> = <ψ| Z⊗I |ψ>  ∈ [-1, 1]."""
    psi = state_vector(x, params, n_layers)
    # Z ⊗ I
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    I = np.eye(2, dtype=np.complex128)
    Z0 = np.kron(Z, I)
    expval = np.real(np.vdot(psi, Z0 @ psi))
    return float(np.clip(expval, -1.0, 1.0))


def batch_expectation(X: np.ndarray, params: np.ndarray, n_layers: int = 3) -> np.ndarray:
    return np.array([expectation_z(x, params, n_layers) for x in X], dtype=np.float64)
