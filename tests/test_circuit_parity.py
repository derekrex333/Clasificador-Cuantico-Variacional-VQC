"""Paridad numpy vs qsharp — requiere pip install .[quantum]."""
import numpy as np
import pytest

from src.quantum.numpy_backend.state import expectation_z
from src.quantum_circuit import circuit_expectation

def test_numpy_expectation_range():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, np.pi, size=2)
    params = rng.uniform(-0.5, 0.5, size=12)
    z = expectation_z(x, params, n_layers=3)
    assert -1.0 - 1e-9 <= z <= 1.0 + 1e-9

def test_numpy_vs_qsharp_parity():
    try:
        import qsharp  # noqa: F401
    except ImportError:
        pytest.skip("qsharp no instalado — pip install .[quantum] para validar paridad")
    rng = np.random.default_rng(1)
    for _ in range(5):
        x = rng.uniform(0, np.pi, size=2)
        params = rng.uniform(-0.5, 0.5, size=12)
        z_numpy = expectation_z(x, params, n_layers=3)
        z_qsharp = circuit_expectation(x, params, n_layers=3, backend="qsharp")
        assert abs(z_numpy - z_qsharp) < 1e-6, f"mismatch {z_numpy} vs {z_qsharp}"
