import numpy as np
from src.quantum.numpy_backend.gates import ry, rz, cnot

def _is_unitary(U):
    return np.allclose(U @ U.conj().T, np.eye(U.shape[0]), atol=1e-10)

def test_ry_unitary():
    assert _is_unitary(ry(0.7))

def test_rz_unitary():
    assert _is_unitary(rz(1.2))

def test_cnot_unitary():
    assert _is_unitary(cnot())
