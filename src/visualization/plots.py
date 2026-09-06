"""Gráficas — fronteras de decisión y curvas de pérdida."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(ax, X: np.ndarray, y: np.ndarray, predict_fn, title: str = ""):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.3, X[:, 0].max() + 0.3
    y_min, y_max = X[:, 1].min() - 0.3, X[:, 1].max() + 0.3
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = predict_fn(grid).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    ax.contour(xx, yy, Z, colors="k", linewidths=0.5, levels=[0.5])
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors="k", s=30)
    ax.set_title(title)
    ax.set_xlabel("feature 0 (→ [0,π])")
    ax.set_ylabel("feature 1 (→ [0,π])")
    return scatter


def plot_comparison(
    X: np.ndarray,
    y: np.ndarray,
    classical_predict,
    quantum_predict,
    save_path: str | None = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    plot_decision_boundary(axes[0], X, y, classical_predict, "Clásico — Regresión Logística")
    plot_decision_boundary(axes[1], X, y, quantum_predict, "Cuántico — VQC (2 qubits, L=3)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def plot_loss_curve(history: dict, save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["loss"], label="train loss")
    if history.get("val_loss") and len(history["val_loss"]) > 0:
        ax.plot(history["val_loss"], label="val loss", linestyle="--")
    ax.set_xlabel("época")
    ax.set_ylabel("BCE loss")
    ax.set_title("Curva de pérdida — VQC")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig
