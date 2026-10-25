"""L2-regularised logistic regression (30 weights + 1 bias = 31 parameters) trained by
mini-batch SGD, plus evaluation metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

N_FEATURES = 30
N_PARAMS = N_FEATURES + 1

LR = 0.05
BATCH = 16
L2 = 1e-3


def init_params() -> np.ndarray:
    return np.zeros(N_PARAMS)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def predict_proba(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return _sigmoid(X @ w[:-1] + w[-1])


def sgd_train(w: np.ndarray, X: np.ndarray, y: np.ndarray, epochs: int, rng: np.random.Generator,
              lr: float = LR, batch: int = BATCH, l2: float = L2) -> np.ndarray:
    w = w.copy()
    n = len(y)
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            b = order[s:s + batch]
            xb, yb = X[b], y[b]
            err = predict_proba(w, xb) - yb
            gw = xb.T @ err / len(b) + l2 * w[:-1]
            gb = err.mean()
            w[:-1] -= lr * gw
            w[-1] -= lr * gb
    return w


def evaluate(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> dict:
    p = predict_proba(w, X)
    yhat = (p >= 0.5).astype(int)
    return {
        "acc": accuracy_score(y, yhat),
        "prec": precision_score(y, yhat, zero_division=0),
        "rec": recall_score(y, yhat, zero_division=0),
        "f1": f1_score(y, yhat, zero_division=0),
        "auc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan"),
    }
