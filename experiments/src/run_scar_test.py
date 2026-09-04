from __future__ import annotations

import argparse, csv, itertools, os, sys, time
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from pu_lib import (PUData, en_estimate_c, evaluate, lee_liu, load_dataset,
                    make_pu, oracle_f1, train_model)
from scar_test import scar_test

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = ["breast", "spambase", "mushroom", "adult", "bank"]
MECHANISMS = ["scar", "sar_attr", "sar_gap", "sar_extreme"]
SEEDS = [0, 1, 2, 3, 4]
CS = [0.3, 0.5]


def standardized(ds, seed):
    X, y = load_dataset(ds, seed=seed)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4,
                                                stratify=y, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5,
                                              stratify=y_tmp, random_state=seed)
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-9
    return ((X_tr - mu) / sd, y_tr, (X_va - mu) / sd, y_va,
            (X_te - mu) / sd, y_te)


def phase_test():
    out = os.path.join(HERE, "results", "scartest_results.csv")
    fields = ["dataset", "mechanism", "c", "seed", "stat", "p", "reject",
              "T0", "null_mean", "purity", "c_hat", "secs"]
    new = not os.path.exists(out)
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        for ds, mech, c, seed in itertools.product(DATASETS, MECHANISMS, CS, SEEDS):
            X_tr, y_tr, *_ = standardized(ds, seed)
            pu = make_pu(X_tr, y_tr, c, mech, seed)
            for stat in ["ks", "nb_auc"]:
                t0 = time.time()
                r = scar_test(X_tr, pu.s, pu.alpha_true, stat=stat, B=200,
                              seed=seed, y_true=y_tr)
                w.writerow(dict(dataset=ds, mechanism=mech, c=c, seed=seed,
                                stat=stat, p=r["p"], reject=r["reject"],
                                T0=round(r["T0"], 5),
                                null_mean=round(r["null_mean"], 5),
                                purity=round(r.get("purity", float("nan")), 4),
                                c_hat=round(r["c_hat"], 4),
                                secs=round(time.time() - t0, 2)))
                fh.flush()
            print(f"[test] {ds} {mech} c={c} seed={seed}", flush=True)
    print(f"done -> {out}")


def phase_methods():
    out = os.path.join(HERE, "results", "e4_results.csv")
    fields = ["phase", "dataset", "mechanism", "c", "seed", "method", "lr", "wd",
              "hidden", "epoch", "alpha_true", "alpha_used",
              "crit_leeliu", "crit_oraclef1", "bacc", "f1", "auc", "secs"]
    grid = [(lr, wd) for lr in (1e-2, 1e-3) for wd in (0.0, 1e-4)]
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        for ds, c, seed in itertools.product(DATASETS, CS, [0, 1, 2]):
            print(f"[methods] {ds} sar_extreme c={c} seed={seed}", flush=True)
            X_tr, y_tr, X_va, y_va, X_te, y_te = standardized(ds, seed)
            pu_tr = make_pu(X_tr, y_tr, c, "sar_extreme", seed)
            pu_va = make_pu(X_va, y_va, c, "sar_extreme", seed + 1000)
            nh = max(1, len(pu_tr.X_p) // 5)
            pu_fit = PUData(pu_tr.X_p[:-nh], pu_tr.X_u, pu_tr.y_u,
                            pu_tr.alpha_true, pu_tr.c, pu_tr.s)
            hold = pu_tr.X_p[-nh:]
            for method in ["uasn", "wce", "en", "upu", "nnpu"]:
                for lr, wd in grid:
                    t0 = time.time()
                    m = train_model(pu_fit, method, pu_tr.alpha_true, lr, wd,
                                    50, seed)
                    thr = 0.5 * en_estimate_c(m, hold) if method == "en" else 0.5
                    w.writerow(dict(
                        phase="e4", dataset=ds, mechanism="sar_extreme", c=c,
                        seed=seed, method=method, lr=lr, wd=wd, hidden=64,
                        epoch=50, alpha_true=round(pu_tr.alpha_true, 4),
                        alpha_used=round(pu_tr.alpha_true, 4),
                        crit_leeliu=round(lee_liu(m, pu_va.X_p, pu_va.X_u, thr), 4),
                        crit_oraclef1=round(oracle_f1(m, X_va, y_va, thr), 4),
                        **{k: round(v, 4) for k, v in
                           evaluate(m, X_te, y_te, thr).items()},
                        secs=round(time.time() - t0, 1)))
                    fh.flush()
    print(f"done -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["test", "methods"])
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    (phase_test if a.phase == "test" else phase_methods)()
