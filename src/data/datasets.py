"""Carga de datasets — moons (default) e iris binario.

Todo se genera en memoria: no hay data/raw/ que versionar.
make_moons y load_iris provienen de scikit-learn y son deterministas
dada la semilla, por lo que la reproducibilidad no depende de archivos.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris, make_moons


def load_moons_dataset(
    n_samples: int = 200,
    noise: float = 0.1,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return X.astype(np.float64), y.astype(np.int64)


def load_iris_binary() -> tuple[np.ndarray, np.ndarray]:
    """Iris reducido a 2 clases (setosa vs versicolor) y 2 features.

    Features: petal length (2) y petal width (3) — las más separables.
    """
    iris = load_iris()
    mask = iris.target < 2  # solo clases 0 y 1
    X = iris.data[mask][:, 2:4]  # petal length, petal width
    y = iris.target[mask]
    return X.astype(np.float64), y.astype(np.int64)


def get_dataset(
    name: str = "moons",
    n_samples: int = 200,
    noise: float = 0.1,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Selector unificado.

    Parameters
    ----------
    name: "moons" | "iris"
    """
    name = name.lower()
    if name == "moons":
        return load_moons_dataset(n_samples=n_samples, noise=noise, random_state=random_state)
    if name == "iris":
        return load_iris_binary()
    raise ValueError(f"dataset desconocido: {name!r} (usar 'moons' o 'iris')")
