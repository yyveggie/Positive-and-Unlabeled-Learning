from __future__ import annotations

import argparse, csv, itertools, os, sys, time
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "methods"))
from pu_lib import (PUData, bbe_alpha, en_estimate_c, evaluate, lee_liu,
                    load_dataset, make_pu, oracle_f1, train_model_v2)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABULAR = ["breast", "spambase", "mushroom", "adult", "bank"]
NEW_METHODS = ["imbpu", "vpu", "distpu", "cvir", "bbe_cvir"]
ALL_METHODS = ["uasn", "wce", "en", "upu", "nnpu", "bbe_nnpu"] + NEW_METHODS
HP_GRID = [(lr, wd) for lr in (1e-2, 1e-3) for wd in (0.0, 1e-4)]

FIELDS = ["phase", "dataset", "mechanism", "c", "seed", "method", "lr", "wd",
          "hidden", "epoch", "alpha_true", "alpha_used",
          "crit_leeliu", "crit_oraclef1", "bacc", "f1", "auc", "secs"]


def prepare(ds, c, seed, max_rows=8000):
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
    return pu_tr, pu_fit, pu_tr.X_p[-n_hold:], pu_va, (X_va, y_va), (X_te, y_te)


def run_cell(writer, fh, phase, ds, c, seed, methods, hp_grid, epochs=50,
             hidden=64, max_rows=8000):
    pu_tr, pu_fit, Xp_hold, pu_va, (X_va, y_va), (X_te, y_te) = \
        prepare(ds, c, seed, max_rows)
    a_bbe = None
    for method in methods:
        base = {"bbe_cvir": "cvir", "bbe_nnpu": "nnpu"}.get(method, method)
        if method.startswith("bbe_") and a_bbe is None:
            a_bbe = bbe_alpha(pu_fit, seed)
        alpha = a_bbe if method.startswith("bbe_") else pu_tr.alpha_true
        for lr, wd in hp_grid:
            t0 = time.time()
            model = train_model_v2(pu_fit, base, alpha, lr, wd, epochs, seed,
                                   hidden=hidden)
            thr = 0.5 * en_estimate_c(model, Xp_hold) if method == "en" else 0.5
            writer.writerow(dict(
                phase=phase, dataset=ds, mechanism="scar", c=c, seed=seed,
                method=method, lr=lr, wd=wd, hidden=hidden, epoch=epochs,
                alpha_true=round(pu_tr.alpha_true, 4), alpha_used=round(alpha, 4),
                crit_leeliu=round(lee_liu(model, pu_va.X_p, pu_va.X_u, thr), 4),
                crit_oraclef1=round(oracle_f1(model, X_va, y_va, thr), 4),
                **{k: round(v, 4) for k, v in
                   evaluate(model, X_te, y_te, thr).items()},
                secs=round(time.time() - t0, 1)))
            fh.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["ext", "mnist", "smoke"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2],
                    help="pass only the new seeds to extend an existing file")
    args = ap.parse_args()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out = os.path.join(HERE, "results", f"{args.phase}_results.csv")
    new = not os.path.exists(out)
    with open(out, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        if args.phase == "smoke":
            run_cell(writer, fh, "smoke", "breast", 0.5, 0, NEW_METHODS,
                     [(1e-2, 0.0)])
        elif args.phase == "ext":
            for ds, c, seed in itertools.product(TABULAR,
                                                 [0.1, 0.3, 0.5, 0.7],
                                                 args.seeds):
                print(f"[ext] {ds} c={c} seed={seed}", flush=True)
                run_cell(writer, fh, "ext", ds, c, seed, NEW_METHODS, HP_GRID)
        else:
            for c, seed in itertools.product([0.3, 0.5], args.seeds):
                print(f"[mnist] c={c} seed={seed}", flush=True)
                run_cell(writer, fh, "mnist", "mnist20k", c, seed, ALL_METHODS,
                         [(1e-3, 0.0), (1e-4, 1e-4)], epochs=30, hidden=128,
                         max_rows=20000)
    print(f"{args.phase} done -> {out}")


if __name__ == "__main__":
    main()
