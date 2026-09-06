import numpy as np
from src.quantum.numpy_backend.state import expectation_z
from src.training.gradients import parameter_shift_grad
from src.training.loss import binary_cross_entropy, z_to_prob

def _finite_diff_grad(params, X, y, n_layers=3, eps=1e-5):
    grad = np.zeros_like(params)
    for i in range(len(params)):
        pp = params.copy(); pp[i] += eps
        pm = params.copy(); pm[i] -= eps
        def loss(p):
            z = np.array([expectation_z(x, p, n_layers) for x in X])
            return binary_cross_entropy(z_to_prob(z), y)
        grad[i] = (loss(pp) - loss(pm)) / (2*eps)
    return grad

def test_parameter_shift_vs_finite_diff():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, np.pi, size=(4, 2))
    y = np.array([0, 1, 0, 1])
    params = rng.uniform(-0.3, 0.3, size=4)  # L=1 para test rápido (4 params)
    g_shift = parameter_shift_grad(params, X, y, n_layers=1)
    g_fd = _finite_diff_grad(params, X, y, n_layers=1)
    assert np.allclose(g_shift, g_fd, atol=1e-4), f"shift {g_shift} vs fd {g_fd}"
