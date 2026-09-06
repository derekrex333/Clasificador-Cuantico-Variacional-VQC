"""Puertas cuánticas como matrices unitarias."""

from __future__ import annotations

import numpy as np


def ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def rz(phi: float) -> np.ndarray:
    return np.array(
        [[np.exp(-1j * phi / 2), 0], [0, np.exp(1j * phi / 2)]],
        dtype=np.complex128,
    )


def cnot() -> np.ndarray:
    """CNOT con control=qubit0, target=qubit1, base |00>,|01>,|10>,|11>."""
    return np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=np.complex128,
    )


def kron(*matrices: np.ndarray) -> np.ndarray:
    out = matrices[0]
    for m in matrices[1:]:
        out = np.kron(out, m)
    return out


def single_qubit_gate_on_state(
    gate: np.ndarray, qubit: int, n_qubits: int
) -> np.ndarray:
    """Expande puerta 2x2 a operador 2^n x 2^n. qubit 0 = más significativo."""
    I = np.eye(2, dtype=np.complex128)
    ops = []
    for q in range(n_qubits):
        ops.append(gate if q == qubit else I)
    return kron(*ops)
