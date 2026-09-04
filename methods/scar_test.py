from __future__ import annotations

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score


def _ks_stat(A: np.ndarray, B: np.ndarray) -> float:
    total = 0.0
    for j in range(A.shape[1]):
        a = np.sort(A[:, j]); b = np.sort(B[:, j])
        grid = np.concatenate([a, b])
        cdf_a = np.searchsorted(a, grid, side="right") / len(a)
        cdf_b = np.searchsorted(b, grid, side="right") / len(b)
        total += float(np.max(np.abs(cdf_a - cdf_b)))
    return total


def _kl_diag(A: np.ndarray, B: np.ndarray, eps: float = 1e-6) -> float:
    m1, m2 = A.mean(0), B.mean(0)
    v1 = A.var(0) + eps
    v2 = B.var(0) + eps
    r = m2 - m1
    return float(0.5 * np.sum(r ** 2 / v1 + v1 / v2 - np.log(v1 / v2) - 1.0))


def _nb_auc(A: np.ndarray, B: np.ndarray, seed: int = 0) -> float:
    X = np.vstack([A, B])
    z = np.concatenate([np.ones(len(A)), -np.ones(len(B))])
    if len(np.unique(z)) < 2:
        return 0.0
    nb = GaussianNB().fit(X, z)
    p = nb.predict_proba(X)[:, list(nb.classes_).index(1.0)]
    return float(abs(roc_auc_score(z, p) - 0.5))


STATS = {"ks": _ks_stat, "kl": _kl_diag, "nb_auc": _nb_auc}


def scar_test(X: np.ndarray, s: np.ndarray, prior: float, stat: str = "ks",
              B: int = 200, seed: int = 0, naive_clf=None,
              y_true: np.ndarray = None) -> dict:
    rng = np.random.default_rng(seed)
    n = len(s)
    L = np.flatnonzero(s == 1)
    U = np.flatnonzero(s == 0)
    if len(L) < 5 or len(U) < 5:
        return {"p": np.nan, "T0": np.nan, "reject": np.nan}


    if naive_clf is None:
        from sklearn.linear_model import LogisticRegression
        naive_clf = LogisticRegression(max_iter=500)
    naive_clf.fit(X, s)
    s_hat = naive_clf.predict_proba(X[U])[:, list(naive_clf.classes_).index(1)]
    c_hat = float(np.clip((len(L) / n) / prior, 1e-3, 1.0))
    k = int(round(n * prior * (1.0 - c_hat)))
    k = int(np.clip(k, 0, len(U)))
    top = U[np.argsort(-s_hat)[:k]]
    P_hat = np.concatenate([L, top])
    if len(P_hat) < 5:
        return {"p": np.nan, "T0": np.nan, "reject": np.nan}

    f = STATS[stat]
    X_Ytilde = X[P_hat]
    T0 = f(X[L], X_Ytilde)


    nulls = np.empty(B)
    for b in range(B):
        draw = rng.random(len(P_hat)) < c_hat
        if draw.sum() < 3:
            draw[rng.choice(len(P_hat), 3, replace=False)] = True
        nulls[b] = f(X[P_hat[draw]], X_Ytilde)

    p = float(np.mean(nulls >= T0))
    out = {"p": p, "T0": float(T0), "null_mean": float(nulls.mean()),
           "reject": int(p < 0.05), "c_hat": c_hat, "k": k}
    if y_true is not None:


        out["purity"] = float(y_true[P_hat].mean())
    return out
