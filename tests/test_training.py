import numpy as np
from src.data.datasets import get_dataset
from src.data.preprocessing import normalize_to_pi, train_test_split_stratified
from src.training.loop import train
from src.training.loss import accuracy_from_probs
from src.training.loop import predict_proba

def test_loss_decreases():
    X_raw, y = get_dataset("moons", n_samples=40, noise=0.1, random_state=0)
    X, _ = normalize_to_pi(X_raw)
    # pocas épocas, solo verificar que no explota y loss no sube descontrolado
    hist = train(X, y, n_layers=2, lr=0.05, epochs=15, optimizer="adam", seed=0)
    assert hist["loss"][-1] <= hist["loss"][0] + 0.1
    assert len(hist["loss"]) == 15

def test_predict_shape():
    X_raw, y = get_dataset("moons", n_samples=20, random_state=1)
    X, _ = normalize_to_pi(X_raw)
    hist = train(X, y, n_layers=2, lr=0.05, epochs=5, seed=1)
    p = predict_proba(X, hist["params_final"], n_layers=2)
    assert p.shape == (20,)
    assert np.all((p >= 0) & (p <= 1))
