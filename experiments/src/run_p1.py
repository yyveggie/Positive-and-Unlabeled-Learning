from __future__ import annotations

import argparse, csv, itertools, os, sys, time
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "methods"))
from pu_lib import (PUData, en_estimate_c, evaluate, lee_liu, load_dataset,
                    make_pu, oracle_f1, tice_alpha, train_model)

METHODS = ["uasn", "wce", "en", "upu", "nnpu", "tice_nnpu"]
HP_GRID = [(lr, wd) for lr in (1e-2, 1e-3) for wd in (0.0, 1e-4)]
EPOCHS = 50

FIELDS = ["dataset", "mechanism", "c", "seed", "method", "lr", "wd",
          "alpha_true", "alpha_used", "crit_leeliu", "crit_oraclef1",
          "bacc", "f1", "auc", "secs"]


def run_config(ds, mech, c, seed, writer, fh):
    X, y = load_dataset(ds, seed=seed)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4,
                                                stratify=y, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5,
                                              stratify=y_tmp, random_state=seed)

    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-9
    X_tr, X_va, X_te = (X_tr - mu) / sd, (X_va - mu) / sd, (X_te - mu) / sd

    pu_tr = make_pu(X_tr, y_tr, c, mech, seed)
    pu_va = make_pu(X_va, y_va, c, mech, seed + 1000)


    n_hold = max(1, len(pu_tr.X_p) // 5)
    Xp_fit, Xp_hold = pu_tr.X_p[:-n_hold], pu_tr.X_p[-n_hold:]
    pu_fit = PUData(Xp_fit, pu_tr.X_u, pu_tr.y_u, pu_tr.alpha_true, pu_tr.c)

    alpha_tice = None
    for method in METHODS:
        if method == "tice_nnpu" and alpha_tice is None:
            alpha_tice = tice_alpha(pu_fit, seed)
        for lr, wd in HP_GRID:
            t0 = time.time()
            base = "nnpu" if method == "tice_nnpu" else method
            alpha = (alpha_tice if method == "tice_nnpu" else pu_tr.alpha_true)
            model = train_model(pu_fit, base, alpha, lr, wd, EPOCHS, seed)

            thr = 0.5
            if method == "en":
                thr = 0.5 * en_estimate_c(model, Xp_hold)

            row = dict(dataset=ds, mechanism=mech, c=c, seed=seed, method=method,
                       lr=lr, wd=wd, alpha_true=round(pu_tr.alpha_true, 4),
                       alpha_used=round(alpha, 4),
                       crit_leeliu=round(lee_liu(model, pu_va.X_p, pu_va.X_u, thr), 4),
                       crit_oraclef1=round(oracle_f1(model, X_va, y_va, thr), 4),
                       **{k: round(v, 4) for k, v in evaluate(model, X_te, y_te, thr).items()},
                       secs=round(time.time() - t0, 1))
            writer.writerow(row); fh.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2],
                    help="pass only the new seeds to extend an existing file")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(here, "results"), exist_ok=True)

    if args.smoke:
        datasets, cs, seeds = ["breast"], [0.5], [0]
        global HP_GRID; HP_GRID = [(1e-2, 0.0)]
        out = args.out or os.path.join(here, "results", "smoke.csv")
    else:
        datasets = args.datasets or ["breast", "spambase", "mushroom", "adult", "bank"]
        cs, seeds = [0.1, 0.3, 0.5, 0.7], args.seeds
        out = args.out or os.path.join(here, "results", "p1_results.csv")

    new = not os.path.exists(out)
    with open(out, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        for ds, c, seed in itertools.product(datasets, cs, seeds):
            print(f"[{time.strftime('%H:%M:%S')}] {ds} scar c={c} seed={seed}", flush=True)
            try:
                run_config(ds, "scar", c, seed, writer, fh)
            except Exception as e:
                print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
