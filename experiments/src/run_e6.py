from __future__ import annotations

import argparse, csv, itertools, os, sys, time
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from pu_lib import (PUData, bbe_alpha, en_estimate_c, evaluate, lee_liu,
                    make_pu, oracle_f1, train_model_v2)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(HERE, "features")
PAIRS = [("cifar", "cifar_raw", "cifar_clip"),
         ("news", "news_tfidf", "news_minilm")]
METHODS = ["uasn", "wce", "en", "upu", "nnpu", "imbpu", "cvir", "bbe_nnpu"]
HP = [(lr, wd) for lr in (1e-2, 1e-3) for wd in (0.0, 1e-4)]
FIELDS = ["task", "representation", "kind", "c", "seed", "method", "lr", "wd",
          "alpha_true", "alpha_used", "crit_leeliu", "crit_oraclef1",
          "bacc", "f1", "auc", "secs"]


def run_cell(w, fh, task, rep, kind, c, seed, epochs=30):
    z = np.load(os.path.join(CACHE, f"{rep}.npz"))
    X, y = z["X"], z["y"]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4,
                                                stratify=y, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5,
                                              stratify=y_tmp, random_state=seed)
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-9
    X_tr, X_va, X_te = (X_tr - mu) / sd, (X_va - mu) / sd, (X_te - mu) / sd
    pu_tr = make_pu(X_tr, y_tr, c, "scar", seed)
    pu_va = make_pu(X_va, y_va, c, "scar", seed + 1000)
    nh = max(1, len(pu_tr.X_p) // 5)
    pu_fit = PUData(pu_tr.X_p[:-nh], pu_tr.X_u, pu_tr.y_u, pu_tr.alpha_true,
                    pu_tr.c, pu_tr.s)
    hold = pu_tr.X_p[-nh:]
    a_bbe = None
    for method in METHODS:
        base = {"bbe_nnpu": "nnpu"}.get(method, method)
        if method.startswith("bbe_") and a_bbe is None:
            a_bbe = bbe_alpha(pu_fit, seed)
        alpha = a_bbe if method.startswith("bbe_") else pu_tr.alpha_true
        for lr, wd in HP:
            t0 = time.time()
            m = train_model_v2(pu_fit, base, alpha, lr, wd, epochs, seed)
            thr = 0.5 * en_estimate_c(m, hold) if method == "en" else 0.5
            w.writerow(dict(
                task=task, representation=rep, kind=kind, c=c, seed=seed,
                method=method, lr=lr, wd=wd,
                alpha_true=round(pu_tr.alpha_true, 4), alpha_used=round(alpha, 4),
                crit_leeliu=round(lee_liu(m, pu_va.X_p, pu_va.X_u, thr), 4),
                crit_oraclef1=round(oracle_f1(m, X_va, y_va, thr), 4),
                **{k: round(v, 4) for k, v in evaluate(m, X_te, y_te, thr).items()},
                secs=round(time.time() - t0, 1)))
            fh.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=["cifar", "news"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2],
                    help="pass only the new seeds to extend an existing file")
    a = ap.parse_args()
    out = os.path.join(HERE, "results", "e6_results.csv")
    new = not os.path.exists(out)
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for task, raw, pre in PAIRS:
            if task not in a.tasks:
                continue
            for rep, kind in [(raw, "raw"), (pre, "pretrained")]:
                if not os.path.exists(os.path.join(CACHE, f"{rep}.npz")):
                    print(f"  missing {rep}.npz, skipped", flush=True)
                    continue
                for c, seed in itertools.product([0.3, 0.5], a.seeds):
                    print(f"[e6] {task}/{kind} c={c} seed={seed}", flush=True)
                    try:
                        run_cell(w, fh, task, rep, kind, c, seed)
                    except Exception as e:
                        print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
