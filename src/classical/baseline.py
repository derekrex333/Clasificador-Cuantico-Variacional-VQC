"""Baseline clásico — regresión logística (+ SVM opcional)."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    **kwargs,
) -> LogisticRegression:
    defaults = dict(max_iter=1000, random_state=42)
    defaults.update(kwargs)
    clf = LogisticRegression(**defaults)
    clf.fit(X_train, y_train)
    return clf


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    kernel: str = "rbf",
    **kwargs,
) -> SVC:
    clf = SVC(kernel=kernel, random_state=42, **kwargs)
    clf.fit(X_train, y_train)
    return clf


def evaluate(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return {"accuracy": float(acc), "y_pred": y_pred}
