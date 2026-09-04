from __future__ import annotations

import argparse, csv, itertools, os, sys, time
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from pu_lib import (PUData, criteria_from_scores, load_dataset, make_pu,
                    metrics_from_scores)
from official_adapters import OFFICIAL_GRID, train_official

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIELDS = ["phase", "dataset", "mechanism", "c", "seed", "method", "cfg",
          "hidden", "alpha_true", "alpha_used", "crit_leeliu", "crit_oraclef1",
          "bacc", "f1", "auc", "auprc", "f1_bestthr", "thr_best",
          "precision", "recall", "pred_rate", "ece", "brier", "secs"]


def prepare(ds, c, seed, max_rows):
    X, y = load_dataset(ds, max_rows=max_rows, seed=seed)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4,
                                                stratify=y, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5,
                                              stratify=y_tmp, random_state=seed)
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-9
    X_tr, X_va, X_te = (X_tr - mu) / sd, (X_va - mu) / sd, (X_te - mu) / sd
    pu_tr = make_pu(X_tr, y_tr, c, "scar", seed)
    pu_va = make_pu(X_va, y_va, c, "scar", seed + 1000)


    n_hold = max(1, len(pu_tr.X_p) // 5)
    pu_fit = PUData(pu_tr.X_p[:-n_hold], pu_tr.X_u, pu_tr.y_u,
                    pu_tr.alpha_true, pu_tr.c)
    return pu_tr, pu_fit, pu_va, (X_va, y_va), (X_te, y_te)


PRIOR_FREE = ("vpu_off", "holistic_off")


def run_cell(w, fh, phase, ds, c, seed, hidden=64, max_rows=8000,
             methods=("vpu_off", "distpu_off")):
    pu_tr, pu_fit, pu_va, (X_va, y_va), (X_te, y_te) = \
        prepare(ds, c, seed, max_rows)
    for method in methods:
        for cfg in OFFICIAL_GRID[method]:
            t0 = time.time()
            try:
                pred = train_official(method, pu_fit, pu_va, pu_tr.alpha_true,
                                      seed, cfg, hidden=hidden)
                crit = criteria_from_scores(pred.proba(pu_va.X_p),
                                            pred.proba(pu_va.X_u))
                w.writerow(dict(
                    phase=phase, dataset=ds, mechanism="scar", c=c, seed=seed,
                    method=method,
                    cfg="|".join(f"{k}{v}" for k, v in cfg.items()),
                    hidden=hidden, alpha_true=round(pu_tr.alpha_true, 4),
                    alpha_used=("" if method in PRIOR_FREE
                                else round(pu_tr.alpha_true, 4)),
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
    ap.add_argument("phase", choices=["smoke", "tab", "mnist"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--datasets", nargs="*",
                    default=["breast", "spambase", "mushroom", "adult", "bank"],
                    help="one per shard when running the grid in parallel")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2],
                    help="pass only the new seeds to extend an existing file")
    ap.add_argument("--methods", nargs="*", default=["vpu_off", "distpu_off"])
    args = ap.parse_args()
    out = args.out or os.path.join(HERE, "results", "officialgrid_results.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    new = not os.path.exists(out)
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        if args.phase == "smoke":
            run_cell(w, fh, "smoke", "breast", 0.5, 0, methods=args.methods)
        elif args.phase == "tab":
            for ds, c, seed in itertools.product(
                    args.datasets, [0.1, 0.3, 0.5, 0.7], args.seeds):
                print(f"[tab] {ds} c={c} seed={seed}", flush=True)
                run_cell(w, fh, "tab", ds, c, seed, methods=args.methods)
        else:
            for c, seed in itertools.product([0.3, 0.5], args.seeds):
                print(f"[mnist] c={c} seed={seed}", flush=True)
                run_cell(w, fh, "mnist", "mnist20k", c, seed,
                         hidden=128, max_rows=20000, methods=args.methods)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
