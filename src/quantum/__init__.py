"""Fachada cuántica — backend numpy por defecto, qsharp opcional."""

from src.quantum.numpy_backend.state import init_params, num_params

__all__ = ["init_params", "num_params"]
