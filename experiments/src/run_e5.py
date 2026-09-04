from __future__ import annotations

import argparse, csv, itertools, os, sys, time
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from pu_lib import (PUData, bbe_alpha, criteria_from_scores, en_estimate_c,
                    lee_liu, load_dataset, make_pu, metrics_from_scores,
                    oracle_f1, scores, subsample_to_alpha, train_model_v2)
from official_adapters import OFFICIAL_GRID, train_official

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METHODS = ["uasn", "wce", "en", "upu", "nnpu", "imbpu", "cvir",
           "bbe_nnpu", "bbe_cvir"]
OFFICIAL = ["vpu_off", "distpu_off", "holistic_off"]
HP = [(lr, wd) for lr in (1e-2, 1e-3) for wd in (0.0, 1e-4)]
FIELDS = ["setting", "dataset", "alpha_target", "c", "seed", "method", "lr", "wd",
          "cfg", "alpha_true", "alpha_used", "crit_leeliu", "crit_oraclef1",
          "bacc", "f1", "auc", "auprc", "f1_bestthr", "thr_best",
          "precision", "recall", "pred_rate", "ece", "brier", "secs"]


def run_cell(w, fh, setting, ds, alpha_target, c, seed, max_rows, epochs,
             methods=METHODS, official=OFFICIAL):
    X, y = load_dataset(ds, max_rows=max_rows, seed=seed)
    if alpha_target is not None:
        X, y = subsample_to_alpha(X, y, alpha_target, seed)
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
    common = dict(setting=setting, dataset=ds, alpha_target=alpha_target, c=c,
                  seed=seed, alpha_true=round(pu_tr.alpha_true, 5))

    for method in methods:
        base = {"bbe_nnpu": "nnpu", "bbe_cvir": "cvir"}.get(method, method)
        if method.startswith("bbe_") and a_bbe is None:
            a_bbe = bbe_alpha(pu_fit, seed)
        alpha = a_bbe if method.startswith("bbe_") else pu_tr.alpha_true
        for lr, wd in HP:
            t0 = time.time()
            m = train_model_v2(pu_fit, base, alpha, lr, wd, epochs, seed)
            thr = 0.5 * en_estimate_c(m, hold) if method == "en" else 0.5
            w.writerow(dict(
                common, method=method, lr=lr, wd=wd, cfg="",
                alpha_used=round(alpha, 5),
                crit_leeliu=round(lee_liu(m, pu_va.X_p, pu_va.X_u, thr), 4),
                crit_oraclef1=round(oracle_f1(m, X_va, y_va, thr), 4),
                **{k: round(v, 4) for k, v in
                   metrics_from_scores(scores(m, X_te), y_te, thr,
                                       extended=True).items()},
                secs=round(time.time() - t0, 1)))
            fh.flush()


    for method in official:
        alpha = float("nan") if method == "vpu_off" else pu_tr.alpha_true
        for cfg in OFFICIAL_GRID[method]:
            t0 = time.time()
            try:
                pred = train_official(method, pu_fit, pu_va, pu_tr.alpha_true,
                                      seed, cfg)
                crit = criteria_from_scores(pred.proba(pu_va.X_p),
                                            pred.proba(pu_va.X_u))
                w.writerow(dict(
                    common, method=method, lr="", wd="",
                    cfg="|".join(f"{k}{v}" for k, v in cfg.items()),
                    alpha_used="" if alpha != alpha else round(alpha, 5),
                    crit_leeliu=round(crit["crit_leeliu"], 4),
                    crit_oraclef1=round(f1_score(y_va, pred.proba(X_va) > 0.5,
                                                 zero_division=0), 4),
                    **{k: round(v, 4) for k, v in
                       metrics_from_scores(pred.proba(X_te), y_te, 0.5,
                                           extended=True).items()},
                    secs=round(time.time() - t0, 1)))
                fh.flush()
            except Exception as e:
                print(f"  FAILED {method} {cfg}: {type(e).__name__}: {e}",
                      flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["sweep", "fraud", "both", "smoke"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--datasets", nargs="*", default=["spambase", "adult"],
                    help="sweep datasets; one per shard when running in parallel")
    ap.add_argument("--methods", nargs="*", default=METHODS)
    ap.add_argument("--official", nargs="*", default=OFFICIAL)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2],
                    help="pass only the new seeds to extend an existing file")
    a = ap.parse_args()
    out = a.out or os.path.join(HERE, "results", "e5_results.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    new = not os.path.exists(out)
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        if a.part == "smoke":
            run_cell(w, fh, "sweep", "breast", 0.10, 0.5, 0, 8000, 50, a.methods, a.official)
        if a.part in ("sweep", "both"):
            for ds, at, c, seed in itertools.product(
                    a.datasets, [0.10, 0.05, 0.02], [0.3, 0.5], a.seeds):
                print(f"[sweep] {ds} alpha={at} c={c} seed={seed}", flush=True)
                try:
                    run_cell(w, fh, "sweep", ds, at, c, seed, 8000, 50, a.methods, a.official)
                except Exception as e:
                    print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
        if a.part in ("fraud", "both"):
            for c, seed in itertools.product([0.3, 0.5], a.seeds):
                print(f"[fraud] creditcard c={c} seed={seed}", flush=True)
                try:
                    run_cell(w, fh, "fraud", "creditcard", None, c, seed, 30000, 30, a.methods, a.official)
                except Exception as e:
                    print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
