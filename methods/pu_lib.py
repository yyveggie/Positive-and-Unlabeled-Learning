from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng


def load_dataset(name: str, max_rows: int = 8000, seed: int = 0):
    if name == "breast":
        d = load_breast_cancer()
        X, y = d.data, d.target
        y = 1 - y
    elif name == "spambase":
        d = fetch_openml("spambase", version=1, as_frame=False, parser="auto")
        X, y = d.data, (d.target == "1").astype(int)
    elif name == "mushroom":
        d = fetch_openml("mushroom", version=1, as_frame=True, parser="auto")
        df = d.frame
        y = (df["class"] == "p").astype(int).to_numpy()
        X = _one_hot(df.drop(columns=["class"]))
    elif name == "adult":
        d = fetch_openml("adult", version=2, as_frame=True, parser="auto")
        df = d.frame.dropna()
        y = (df["class"] == ">50K").astype(int).to_numpy()
        X = _one_hot(df.drop(columns=["class"]))
    elif name == "creditcard":


        d = fetch_openml("creditcard", version=1, as_frame=False, parser="auto")
        X = np.asarray(d.data, dtype=np.float64)
        y = (np.asarray(d.target).astype(str) == "1").astype(int)
        if y.sum() == 0:
            y = (np.asarray(d.target).astype(str).str.strip("'") == "1").astype(int)
    elif name == "mnist20k":
        d = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X = d.data / 255.0
        y = (d.target.astype(int) % 2 == 0).astype(int)
        max_rows = min(max_rows, 20000)
    elif name == "bank":
        d = fetch_openml("bank-marketing", version=1, as_frame=True, parser="auto")
        df = d.frame
        target = df.columns[-1]
        y = (df[target].astype(str).isin(["2", "yes"])).astype(int).to_numpy()
        X = _one_hot(df.drop(columns=[target]))
    else:
        raise ValueError(name)
    X = np.asarray(X, dtype=np.float64)
    if len(y) > max_rows:
        idx = RNG(seed).choice(len(y), max_rows, replace=False)
        X, y = X[idx], y[idx]
    return X, np.asarray(y, dtype=int)


def _one_hot(df):
    import pandas as pd
    num = df.select_dtypes(include=[np.number])
    cat = df.select_dtypes(exclude=[np.number])
    parts = [num.to_numpy(dtype=np.float64)] if num.shape[1] else []
    if cat.shape[1]:
        parts.append(pd.get_dummies(cat, drop_first=True).to_numpy(dtype=np.float64))
    return np.hstack(parts)


@dataclass
class PUData:
    X_p: np.ndarray
    X_u: np.ndarray
    y_u: np.ndarray
    alpha_true: float
    c: float
    s: np.ndarray = None
    e_true: np.ndarray = None


def subsample_to_alpha(X, y, target_alpha: float, seed: int):
    rng = RNG(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    n_pos = int(round(target_alpha * len(neg) / max(1 - target_alpha, 1e-9)))
    if n_pos >= len(pos):
        return X, y
    keep = np.concatenate([neg, rng.choice(pos, n_pos, replace=False)])
    keep.sort()
    return X[keep], y[keep]


def driving_feature(X, y, mode: str = "max") -> int:
    corr = np.array([abs(np.corrcoef(X[:, j], y)[0, 1]) if X[:, j].std() > 0 else 0
                     for j in range(X.shape[1])])
    if mode == "min":
        corr = np.where(X.std(0) > 0, corr, np.inf)
        return int(np.argmin(corr))
    return int(np.argmax(corr))


def make_pu(X, y, c: float, mechanism: str, seed: int) -> PUData:
    rng = RNG(seed)
    pos_idx = np.flatnonzero(y == 1)
    n_label = max(1, int(round(c * len(pos_idx))))

    if mechanism == "scar":
        w = np.ones(len(pos_idx))
    elif mechanism == "sar_attr":

        corr = np.array([abs(np.corrcoef(X[:, j], y)[0, 1]) if X[:, j].std() > 0 else 0
                         for j in range(X.shape[1])])
        j = int(np.argmax(corr))
        z = (X[pos_idx, j] - X[pos_idx, j].mean()) / (X[pos_idx, j].std() + 1e-9)
        w = 1.0 / (1.0 + np.exp(-2.0 * z))
    elif mechanism == "sar_gap":

        lr = LogisticRegression(max_iter=1000).fit(X, y)
        p = lr.predict_proba(X[pos_idx])[:, 1]
        w = p ** 2
    elif mechanism == "sar_indep":


        j = driving_feature(X, y, mode="min")
        z = (X[pos_idx, j] - X[pos_idx, j].mean()) / (X[pos_idx, j].std() + 1e-9)
        w = 1.0 / (1.0 + np.exp(-2.0 * z))
    elif mechanism == "sar_extreme":


        corr = np.array([abs(np.corrcoef(X[:, j], y)[0, 1]) if X[:, j].std() > 0 else 0
                         for j in range(X.shape[1])])
        j = int(np.argmax(corr))
        z = (X[pos_idx, j] - X[pos_idx, j].mean()) / (X[pos_idx, j].std() + 1e-9)
        w = 1.0 / (1.0 + np.exp(-4.0 * z)) ** 3 + 0.02
    else:
        raise ValueError(mechanism)

    if mechanism == "scar":


        labeled = rng.choice(pos_idx, size=n_label, replace=False, p=w / w.sum())
        e = np.full(len(pos_idx), n_label / len(pos_idx), dtype=float)
    else:


        e = np.clip(w * c / w.mean(), 0.0, 1.0)
        drawn = rng.random(len(pos_idx)) < e
        if drawn.sum() < 5:
            drawn[rng.choice(len(pos_idx), 5, replace=False)] = True
        labeled = pos_idx[drawn]

    s = np.zeros(len(X), dtype=int)
    s[labeled] = 1
    e_all = np.full(len(X), np.nan, dtype=float)
    e_all[pos_idx] = e
    return PUData(X_p=X[labeled], X_u=X, y_u=y, alpha_true=float(y.mean()),
                  c=len(labeled) / len(pos_idx), s=s, e_true=e_all)


class MLP(nn.Module):
    def __init__(self, d_in, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_model_ckpt(pu: PUData, method: str, alpha: float, lr: float, wd: float,
                     seed: int, ckpts=(10, 25, 50), hidden=64, device="cpu"):
    import copy
    torch.manual_seed(seed)
    Xp = torch.tensor(pu.X_p, dtype=torch.float32, device=device)
    Xu = torch.tensor(pu.X_u, dtype=torch.float32, device=device)
    model = MLP(Xp.shape[1], hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    n_p, n_u = len(Xp), len(Xu)
    batch, snaps = 512, {}

    for epoch in range(1, max(ckpts) + 1):
        perm_u = torch.randperm(n_u, device=device)
        for start in range(0, n_u, batch):
            iu = perm_u[start:start + batch]
            ip = torch.randint(0, n_p, (min(batch, n_p),), device=device)
            zp, zu = model(Xp[ip]), model(Xu[iu])
            loss = _pu_loss(method, alpha, zp, zu, n_p, n_u)
            opt.zero_grad(); loss.backward(); opt.step()
        if epoch in ckpts:
            snaps[epoch] = copy.deepcopy(model)
    return snaps


def _pu_loss(method, alpha, zp, zu, n_p, n_u):
    if method in ("uasn", "wce", "en"):
        w_pos = n_u / (n_p + n_u) if method == "wce" else 0.5
        return (w_pos * nn.functional.softplus(-zp).mean()
                + (1 - w_pos) * nn.functional.softplus(zu).mean())
    if method == "upu":
        return (alpha * _sigmoid_loss(zp).mean()
                - alpha * _sigmoid_loss(-zp).mean()
                + _sigmoid_loss(-zu).mean())
    if method == "nnpu":
        r_neg = _sigmoid_loss(-zu).mean() - alpha * _sigmoid_loss(-zp).mean()
        loss = alpha * _sigmoid_loss(zp).mean() + torch.clamp(r_neg, min=0.0)
        return -r_neg if r_neg < 0 else loss
    if method == "imbpu":


        pi_re = 0.5
        r_neg = _sigmoid_loss(-zu).mean() - alpha * _sigmoid_loss(-zp).mean()
        if r_neg < 0:
            return -r_neg
        return pi_re * _sigmoid_loss(zp).mean() \
            + (1 - pi_re) * r_neg / max(1 - alpha, 1e-3)
    if method == "distpu":


        sig_u = torch.sigmoid(zu).clamp(1e-6, 1 - 1e-6)
        ent = -(sig_u * sig_u.log() + (1 - sig_u) * (1 - sig_u).log()).mean()
        return (nn.functional.softplus(-zp).mean()
                + (sig_u.mean() - alpha).abs()
                + 0.1 * ent)
    raise ValueError(method)


def train_model_v2(pu: PUData, method: str, alpha: float, lr: float, wd: float,
                   epochs: int, seed: int, hidden=64, device="cpu") -> nn.Module:
    torch.manual_seed(seed)
    Xp = torch.tensor(pu.X_p, dtype=torch.float32, device=device)
    Xu = torch.tensor(pu.X_u, dtype=torch.float32, device=device)
    model = MLP(Xp.shape[1], hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    n_p, n_u = len(Xp), len(Xu)
    batch = 512
    beta = torch.distributions.Beta(0.3, 0.3)


    dc_omega, dc_theta = 0.0, 0.0
    dc_beta, dc_tau, dc_gamma = 0.45, 2e-3, 0.0

    for _ in range(epochs):
        if method == "cvir":
            with torch.no_grad():
                su = model(Xu)
            k = max(1, int((1 - alpha) * n_u))
            keep = torch.argsort(su)[:k]
            pool = keep[torch.randperm(len(keep), device=device)]
        else:
            pool = torch.randperm(n_u, device=device)

        for start in range(0, len(pool), batch):
            iu = pool[start:start + batch]
            ip = torch.randint(0, n_p, (min(batch, n_p),), device=device)
            zp, zu = model(Xp[ip]), model(Xu[iu])

            if method == "vpu":
                gp = torch.sigmoid(zp).clamp(1e-6, 1.0)
                gu = torch.sigmoid(zu).clamp(1e-6, 1.0)
                loss = gu.mean().log() - gp.log().mean()
                lam = beta.sample().to(device)
                m = min(len(ip), len(iu))
                x_mix = lam * Xu[iu[:m]] + (1 - lam) * Xp[ip[:m]]
                t_mix = (lam * gu[:m].detach() + (1 - lam) * 1.0).clamp(1e-6, 1.0)
                g_mix = torch.sigmoid(model(x_mix)).clamp(1e-6, 1.0)
                loss = loss + 0.3 * ((g_mix.log() - t_mix.log()) ** 2).mean()
            elif method == "cvir":
                loss = (nn.functional.softplus(-zp).mean()
                        + nn.functional.softplus(zu).mean())
            elif method == "dcpu":

                r_p_pos = _sigmoid_loss(zp).mean()
                r_p_neg = _sigmoid_loss(-zp).mean()
                r_u_neg = _sigmoid_loss(-zu).mean()
                neg = r_u_neg - alpha * r_p_neg
                dc_omega = (dc_beta * dc_omega
                            + (1 - dc_beta) * (1 - alpha) * float(r_p_pos.detach()))
                resid = r_p_pos - neg / max(1 - alpha, 1e-3)
                if float(neg.detach()) >= dc_omega - dc_gamma:
                    loss = (alpha * r_p_pos
                            + torch.clamp(neg, min=dc_omega)
                            + 0.5 * dc_tau * (resid + dc_theta / dc_tau) ** 2)
                else:
                    loss = -neg
                dc_theta = dc_theta + dc_tau * float(resid.detach())
            else:
                loss = _pu_loss(method, alpha, zp, zu, n_p, n_u)

            opt.zero_grad(); loss.backward(); opt.step()
    return model


def _sigmoid_loss(z):
    return torch.sigmoid(-z)


def train_model(pu: PUData, method: str, alpha: float, lr: float, wd: float,
                epochs: int, seed: int, device="cpu") -> nn.Module:
    torch.manual_seed(seed)
    Xp = torch.tensor(pu.X_p, dtype=torch.float32, device=device)
    Xu = torch.tensor(pu.X_u, dtype=torch.float32, device=device)
    model = MLP(Xp.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    n_p, n_u = len(Xp), len(Xu)
    batch = 512

    for _ in range(epochs):
        perm_u = torch.randperm(n_u, device=device)
        for start in range(0, n_u, batch):
            iu = perm_u[start:start + batch]
            ip = torch.randint(0, n_p, (min(batch, n_p),), device=device)
            zp, zu = model(Xp[ip]), model(Xu[iu])

            if method in ("uasn", "wce", "en"):
                w_pos = n_u / (n_p + n_u) if method == "wce" else 0.5
                loss = (w_pos * nn.functional.softplus(-zp).mean()
                        + (1 - w_pos) * nn.functional.softplus(zu).mean())
            elif method == "upu":
                loss = (alpha * _sigmoid_loss(zp).mean()
                        - alpha * _sigmoid_loss(-zp).mean()
                        + _sigmoid_loss(-zu).mean())
            elif method == "nnpu":
                r_neg = _sigmoid_loss(-zu).mean() - alpha * _sigmoid_loss(-zp).mean()
                loss = alpha * _sigmoid_loss(zp).mean() + torch.clamp(r_neg, min=0.0)
                if r_neg < 0:
                    loss = -r_neg
            else:
                raise ValueError(method)

            opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def scores(model, X, device="cpu"):
    z = model(torch.tensor(X, dtype=torch.float32, device=device))
    return torch.sigmoid(z).cpu().numpy()


def en_estimate_c(model, X_p_holdout) -> float:
    return float(np.clip(scores(model, X_p_holdout).mean(), 1e-3, 1.0))


def tice_alpha(pu: PUData, seed: int, delta=0.05, depth=3, min_node=32) -> float:
    X = np.vstack([pu.X_p, pu.X_u])
    s = np.concatenate([np.ones(len(pu.X_p)), np.zeros(len(pu.X_u))])
    rng = RNG(seed)
    idx = rng.permutation(len(s))
    half = len(s) // 2
    folds = [(idx[:half], idx[half:]), (idx[half:], idx[:half])]

    def lb(mask_est):
        T = mask_est.sum()
        if T < min_node:
            return -1.0
        L = s[mask_est].sum()
        return L / T - np.sqrt(np.log(1 / delta) / (2 * T))

    def search(tree_mask, est_mask, d):
        best = lb(est_mask)
        if d == 0:
            return best
        n_feat = X.shape[1]
        for j in rng.choice(n_feat, size=min(10, n_feat), replace=False):
            vals = X[tree_mask, j]
            if len(vals) < 2 * min_node:
                continue
            thr = np.quantile(vals, [0.25, 0.5, 0.75])
            for t in np.unique(thr):
                for side_mask_fn in (lambda a: a <= t, lambda a: a > t):
                    tm = tree_mask & side_mask_fn(X[:, j])
                    em = est_mask & side_mask_fn(X[:, j])
                    if s[tm].sum() / max(tm.sum(), 1) > s[tree_mask].sum() / max(tree_mask.sum(), 1):
                        best = max(best, search(tm, em, d - 1))
        return best

    ests = []
    for tr, es in folds:
        m_tr = np.zeros(len(s), bool); m_tr[tr] = True
        m_es = np.zeros(len(s), bool); m_es[es] = True
        ests.append(max(lb(m_es), search(m_tr, m_es, depth)))
    c_mix = float(np.clip(np.mean(ests), 1e-3, 0.999))

    L, M = len(pu.X_p), len(pu.X_u)
    return float(np.clip(L * (1 - c_mix) / (c_mix * M), 1e-3, 1.0))


def bbe_alpha(pu: PUData, seed: int, delta: float = 0.1) -> float:
    rng = RNG(seed)
    n_p, n_u = len(pu.X_p), len(pu.X_u)
    ip, iu = rng.permutation(n_p), rng.permutation(n_u)
    cut_p, cut_u = max(2, n_p // 2), max(2, n_u // 2)
    fit = PUData(pu.X_p[ip[cut_p:]], pu.X_u[iu[cut_u:]], pu.y_u[iu[cut_u:]],
                 pu.alpha_true, pu.c)
    model = train_model(fit, "uasn", 0.0, 1e-2, 0.0, 100, seed)
    sp = scores(model, pu.X_p[ip[:cut_p]])
    su = scores(model, pu.X_u[iu[:cut_u]])
    pen = np.sqrt(np.log(1 / delta) / 2) * (1 / np.sqrt(len(sp)) + 1 / np.sqrt(len(su)))
    min_pos = max(3, int(0.02 * len(sp)))
    best_bound, best_ratio = np.inf, 1.0
    for z in np.unique(np.quantile(np.concatenate([sp, su]),
                                   np.linspace(0.05, 0.999, 96))):
        qp = (sp >= z).mean()
        if qp * len(sp) < min_pos:
            continue
        ratio = (su >= z).mean() / qp
        bound = ratio + pen / qp
        if bound < best_bound:
            best_bound, best_ratio = bound, ratio
    return float(np.clip(best_ratio, 1e-3, 1.0))


def criteria_from_scores(sp_val, su_val, alpha=None, threshold=0.5) -> dict:
    sp = np.asarray(sp_val, dtype=np.float64)
    su = np.asarray(su_val, dtype=np.float64)
    r = float((sp > threshold).mean())
    pr1 = float((su > threshold).mean())
    out = {"crit_leeliu": r * r / max(pr1, 1e-6),
           "crit_puauc": float(roc_auc_score(
               np.concatenate([np.ones(len(sp)), np.zeros(len(su))]),
               np.concatenate([sp, su])))}
    if alpha is not None and alpha == alpha:
        risk = alpha * (1.0 - sp).mean() + max(0.0, su.mean() - alpha * sp.mean())
        out["crit_nnpurisk"] = -float(risk)
        out["crit_f1hat"] = 2.0 * alpha * r / max(alpha + pr1, 1e-9)
    else:
        out["crit_nnpurisk"] = float("nan")
        out["crit_f1hat"] = float("nan")
    return out


def lee_liu(model, Xp_val, Xu_val, threshold=0.5) -> float:
    r = float((scores(model, Xp_val) > threshold).mean())
    pr1 = float((scores(model, Xu_val) > threshold).mean())
    return r * r / max(pr1, 1e-6)


def all_criteria(model, Xp_val, Xu_val, alpha=None, threshold=0.5) -> dict:
    return criteria_from_scores(scores(model, Xp_val), scores(model, Xu_val),
                                alpha, threshold)


def oracle_f1(model, X_val, y_val, threshold=0.5) -> float:
    return f1_score(y_val, scores(model, X_val) > threshold, zero_division=0)


def expected_calibration_error(p, y, n_bins: int = 15) -> float:
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    idx = np.clip(np.digitize(p, np.linspace(0, 1, n_bins + 1)[1:-1]), 0, n_bins - 1)
    err = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            err += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(err)


def metrics_from_scores(p, y_true, threshold=0.5, extended=False) -> dict:
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.int64)
    yhat = p > threshold
    two_class = len(np.unique(y)) > 1
    out = {"bacc": balanced_accuracy_score(y, yhat),
           "f1": f1_score(y, yhat, zero_division=0),
           "auc": roc_auc_score(y, p) if two_class else float("nan")}
    if not extended:
        return out
    if two_class:
        prec, rec, thr = precision_recall_curve(y, p)
        f1s = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
        i = int(np.nanargmax(f1s))
        out.update(auprc=average_precision_score(y, p),
                   f1_bestthr=float(f1s[i]),
                   thr_best=float(thr[i]) if i < len(thr) else 1.0)
    else:
        out.update(auprc=float("nan"), f1_bestthr=float("nan"),
                   thr_best=float("nan"))
    rec = recall_score(y, yhat, zero_division=0)
    neg = (y == 0)
    fpr = float(yhat[neg].mean()) if neg.any() else float("nan")
    out.update(precision=precision_score(y, yhat, zero_division=0),
               recall=rec,
               fnr=float(1.0 - rec),
               fpr=fpr,
               gap=abs(float(1.0 - rec) - fpr) if neg.any() else float("nan"),
               pred_rate=float(yhat.mean()),
               ece=expected_calibration_error(p, y),
               brier=float(np.mean((p - y) ** 2)))
    return out


def evaluate(model, X_test, y_test, threshold=0.5, extended=False) -> dict:
    return metrics_from_scores(scores(model, X_test), y_test, threshold, extended)
