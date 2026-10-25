"""Wisconsin Diagnostic Breast Cancer data, partitioning and federated standardisation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

HOSPITALS = ["Hospital A", "Hospital B", "Hospital C"]
IID_RATIO = (2, 4, 3)


@dataclass
class ClientData:
    name: str
    X: np.ndarray
    y: np.ndarray

    @property
    def n(self) -> int:
        return len(self.y)


def load_wdbc():
    d = load_breast_cancer()
    X = d.data.astype(np.float64)
    y = (d.target == 0).astype(np.int64)  # malignant (target 0 in sklearn) = positive class
    return X, y


def split_train_test(seed: int, test_size: float = 0.2):
    X, y = load_wdbc()
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def partition_iid(X, y, seed: int, ratio=IID_RATIO) -> list[ClientData]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    total = sum(ratio)
    sizes = [len(y) * r // total for r in ratio[:-1]]
    sizes.append(len(y) - sum(sizes))
    out, start = [], 0
    for name, s in zip(HOSPITALS, sizes):
        part = idx[start:start + s]
        out.append(ClientData(name, X[part], y[part]))
        start += s
    return out


def partition_dirichlet(X, y, seed: int, alpha: float = 0.5, min_size: int = 20,
                        n_clients: int = 3) -> list[ClientData]:
    """Label-skewed split: per-class proportions drawn from Dir(alpha) [Hsu et al. 2019].
    Draws are repeated until every client holds at least ``min_size`` cases."""
    rng = np.random.default_rng(seed + 10_000)
    classes = np.unique(y)
    while True:
        parts = [[] for _ in range(n_clients)]
        for c in classes:
            idx = rng.permutation(np.where(y == c)[0])
            p = rng.dirichlet(alpha * np.ones(n_clients))
            cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(idx, cuts)):
                parts[k].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            break
    return [ClientData(HOSPITALS[k], X[np.array(p)], y[np.array(p)]) for k, p in enumerate(parts)]


def federated_standardise(clients: list[ClientData], X_test: np.ndarray):
    """Global mean/variance from per-hospital counts, sums and sums of squares only."""
    n = sum(c.n for c in clients)
    s = sum(c.X.sum(axis=0) for c in clients)
    ss = sum((c.X ** 2).sum(axis=0) for c in clients)
    mean = s / n
    var = ss / n - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-12))
    scaled = [ClientData(c.name, (c.X - mean) / std, c.y.copy()) for c in clients]
    return scaled, (X_test - mean) / std


def make_partitions(seed: int, setting: str):
    X_tr, X_te, y_tr, y_te = split_train_test(seed)
    if setting == "iid":
        clients = partition_iid(X_tr, y_tr, seed)
    elif setting == "noniid":
        clients = partition_dirichlet(X_tr, y_tr, seed)
    else:
        raise ValueError(setting)
    clients, X_te = federated_standardise(clients, X_te)
    return clients, X_te, y_te
