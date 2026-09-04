from __future__ import annotations

import argparse, csv, itertools, os, sys, tarfile, time, urllib.request

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "methods"))
from pu_lib import _pu_loss, criteria_from_scores, metrics_from_scores

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(HERE, "features", "graph")
RNG = np.random.default_rng

DATASETS = ["cora", "citeseer"]
REWIRE = [0.0, 0.4, 0.8]
BACKBONES = ["mlp", "gcn"]
METHODS = ["uasn", "wce", "nnpu", "imbpu", "cvir", "bbe_nnpu"]
HP = [(lr, wd) for lr in (1e-2, 1e-3) for wd in (0.0, 5e-4)]
EPOCHS = 100
FIELDS = ["dataset", "rewire", "homophily", "backbone", "c", "seed", "method",
          "lr", "wd", "alpha_true", "alpha_used", "crit_leeliu", "crit_oraclef1",
          "bacc", "f1", "auc", "auprc", "f1_bestthr", "thr_best",
          "precision", "recall", "pred_rate", "ece", "brier", "secs"]


def load_graph(name):
    d = os.path.join(CACHE, name)
    if not os.path.isdir(d):
        os.makedirs(CACHE, exist_ok=True)
        tgz = os.path.join(CACHE, name + ".tgz")
        urllib.request.urlretrieve(
            f"https://linqs-data.soe.ucsc.edu/public/lbc/{name}.tgz", tgz)
        with tarfile.open(tgz) as t:
            t.extractall(CACHE)
    ids, feats, labels = [], [], []
    with open(os.path.join(d, f"{name}.content"), encoding="utf8") as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            ids.append(p[0]); feats.append([int(v) for v in p[1:-1]]); labels.append(p[-1])
    idx = {k: i for i, k in enumerate(ids)}
    classes = sorted(set(labels))
    y = np.array([classes.index(l) for l in labels], dtype=int)
    X = np.asarray(feats, dtype=np.float32)
    edges = []
    with open(os.path.join(d, f"{name}.cites"), encoding="utf8") as fh:
        for line in fh:
            a, b = line.split()
            if a in idx and b in idx and a != b:
                edges.append((idx[a], idx[b]))
    return X, y, np.unique(np.asarray(edges), axis=0)


def rewire(edges, y, frac, seed):
    if frac <= 0:
        return edges
    rng = RNG(seed)
    e = edges.copy()
    pick = rng.random(len(e)) < frac
    by_class = {c: np.flatnonzero(y != c) for c in np.unique(y)}
    for i in np.flatnonzero(pick):
        pool = by_class[y[e[i, 0]]]
        e[i, 1] = pool[rng.integers(len(pool))]
    return e


def homophily(edges, y):
    return float((y[edges[:, 0]] == y[edges[:, 1]]).mean()) if len(edges) else 0.0


def norm_adj(edges, n, device="cpu"):
    src = np.concatenate([edges[:, 0], edges[:, 1], np.arange(n)])
    dst = np.concatenate([edges[:, 1], edges[:, 0], np.arange(n)])
    deg = np.bincount(src, minlength=n).astype(np.float32)
    w = 1.0 / np.sqrt(np.maximum(deg[src], 1) * np.maximum(deg[dst], 1))
    i = torch.tensor(np.vstack([src, dst]), dtype=torch.long, device=device)
    return torch.sparse_coo_tensor(i, torch.tensor(w, device=device),
                                   (n, n)).coalesce()


class Net(nn.Module):

    def __init__(self, d_in, hidden=64, dropout=0.5):
        super().__init__()
        self.l1, self.l2 = nn.Linear(d_in, hidden), nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, X, A=None):
        h = self.l1(self.drop(X))
        if A is not None:
            h = torch.sparse.mm(A, h)
        h = torch.relu(h)
        h = self.l2(self.drop(h))
        if A is not None:
            h = torch.sparse.mm(A, h)
        return h.squeeze(-1)


def train_graph_pu(X, A, p_idx, u_idx, method, alpha, lr, wd, seed, epochs=EPOCHS):
    torch.manual_seed(seed)
    model = Net(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    keep = u_idx
    for ep in range(epochs):
        if method == "cvir" and ep and ep % 10 == 0:
            model.eval()
            with torch.no_grad():
                su = torch.sigmoid(model(X, A))[u_idx]
            k = max(1, int((1 - alpha) * len(u_idx)))
            keep = u_idx[torch.argsort(su)[:k]]
        model.train()
        z = model(X, A)
        if method == "cvir":
            loss = (nn.functional.softplus(-z[p_idx]).mean()
                    + nn.functional.softplus(z[keep]).mean())
        else:
            loss = _pu_loss(method, alpha, z[p_idx], z[keep], len(p_idx), len(keep))
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


@torch.no_grad()
def scores_of(model, X, A):
    return torch.sigmoid(model(X, A)).numpy()


def bbe_prior(X, A, p_idx, u_idx, seed, delta=0.1):
    rng = RNG(seed)
    pi = p_idx[torch.tensor(rng.permutation(len(p_idx)))]
    ui = u_idx[torch.tensor(rng.permutation(len(u_idx)))]
    cp, cu = max(2, len(pi) // 2), max(2, len(ui) // 2)
    m = train_graph_pu(X, A, pi[cp:], ui[cu:], "uasn", 0.0, 1e-2, 5e-4, seed, 100)
    s = scores_of(m, X, A)
    sp, su = s[pi[:cp].numpy()], s[ui[:cu].numpy()]
    pen = np.sqrt(np.log(1 / delta) / 2) * (1 / np.sqrt(len(sp)) + 1 / np.sqrt(len(su)))
    best_b, best_r = np.inf, 1.0
    for z in np.unique(np.quantile(np.concatenate([sp, su]), np.linspace(0.05, 0.999, 96))):
        qp = (sp >= z).mean()
        if qp * len(sp) < max(3, 0.02 * len(sp)):
            continue
        r = (su >= z).mean() / qp
        if r + pen / qp < best_b:
            best_b, best_r = r + pen / qp, r
    return float(np.clip(best_r, 1e-3, 1.0))


def run_cell(w, fh, ds, frac, backbone, c, seed, cache):
    X0, y0, e0 = cache[ds]
    n = len(y0)
    pos_class = int(np.bincount(y0).argmax())
    y = (y0 == pos_class).astype(int)
    edges = rewire(e0, y0, frac, seed)
    h = homophily(edges, y0)
    X = torch.tensor(X0)
    A = norm_adj(edges, n) if backbone == "gcn" else None

    rng = RNG(seed)
    perm = rng.permutation(n)
    n_tr, n_va = int(0.6 * n), int(0.2 * n)
    tr, va, te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]
    pos_tr = tr[y[tr] == 1]
    lab = pos_tr[rng.random(len(pos_tr)) < c]
    if len(lab) < 5:
        lab = pos_tr[:5]
    p_idx = torch.tensor(np.sort(lab), dtype=torch.long)
    u_idx = torch.tensor(np.sort(tr), dtype=torch.long)
    pv = va[y[va] == 1]
    pv = pv[rng.random(len(pv)) < c]
    if len(pv) < 5:
        pv = va[y[va] == 1][:5]
    alpha_true = float(y[tr].mean())
    a_bbe = None

    for method in METHODS:
        base = "nnpu" if method == "bbe_nnpu" else method
        if method == "bbe_nnpu" and a_bbe is None:
            a_bbe = bbe_prior(X, A, p_idx, u_idx, seed)
        alpha = a_bbe if method == "bbe_nnpu" else alpha_true
        for lr, wd in HP:
            t0 = time.time()
            m = train_graph_pu(X, A, p_idx, u_idx, base, alpha, lr, wd, seed)
            s = scores_of(m, X, A)
            crit = criteria_from_scores(s[pv], s[va])
            w.writerow(dict(
                dataset=ds, rewire=frac, homophily=round(h, 4), backbone=backbone,
                c=c, seed=seed, method=method, lr=lr, wd=wd,
                alpha_true=round(alpha_true, 4), alpha_used=round(alpha, 4),
                crit_leeliu=round(crit["crit_leeliu"], 4),
                crit_oraclef1=round(metrics_from_scores(s[va], y[va])["f1"], 4),
                **{k: round(v, 4) for k, v in
                   metrics_from_scores(s[te], y[te], 0.5, extended=True).items()},
                secs=round(time.time() - t0, 1)))
            fh.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--rewire", nargs="*", type=float, default=REWIRE)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(HERE, "results", "graph_results.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cache = {d: load_graph(d) for d in a.datasets}
    for d, (X, y, e) in cache.items():
        print(f"[graph] {d}: {len(y)} nodes, {len(e)} edges, {X.shape[1]} features, "
              f"homophily {homophily(e, y):.3f}, largest class "
              f"{np.bincount(y).max() / len(y):.3f}", flush=True)
    cells = ([(a.datasets[0], 0.0, "gcn", 0.5, 0)] if a.smoke else
             list(itertools.product(a.datasets, a.rewire, BACKBONES,
                                    [0.1, 0.5], a.seeds)))
    new = not os.path.exists(out)
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for ds, frac, bb, c, seed in cells:
            print(f"[graph] {ds} rewire={frac} {bb} c={c} seed={seed}", flush=True)
            try:
                run_cell(w, fh, ds, frac, bb, c, seed, cache)
            except Exception as e:
                print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
