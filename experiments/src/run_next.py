from __future__ import annotations

import argparse, csv, itertools, os, sys, time
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from pu_lib import (PUData, bbe_alpha, en_estimate_c, evaluate, lee_liu,
                    load_dataset, make_pu, oracle_f1, scores, train_model,
                    train_model_ckpt)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = ["breast", "spambase", "mushroom", "adult", "bank"]
HP_GRID = [(lr, wd) for lr in (1e-2, 1e-3) for wd in (0.0, 1e-4)]
SEEDS = [0, 1, 2]

FIELDS = ["phase", "dataset", "mechanism", "c", "seed", "method", "lr", "wd",
          "hidden", "epoch", "alpha_true", "alpha_used",
          "crit_leeliu", "crit_oraclef1", "bacc", "f1", "auc", "secs"]


def prepare(ds, mech, c, seed):
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
    pu_fit = PUData(pu_tr.X_p[:-n_hold], pu_tr.X_u, pu_tr.y_u,
                    pu_tr.alpha_true, pu_tr.c)
    return pu_tr, pu_fit, pu_tr.X_p[-n_hold:], pu_va, (X_va, y_va), (X_te, y_te)


def emit(writer, fh, **kw):
    row = {f: kw.get(f, "") for f in FIELDS}
    writer.writerow(row); fh.flush()


def fit_and_score(pu_fit, base, alpha, lr, wd, seed, Xp_hold, pu_va, val, test,
                  method):
    t0 = time.time()
    model = train_model(pu_fit, base, alpha, lr, wd, 50, seed)
    thr = 0.5 * en_estimate_c(model, Xp_hold) if method == "en" else 0.5
    (X_va, y_va), (X_te, y_te) = val, test
    return dict(lr=lr, wd=wd, hidden=64, epoch=50,
                alpha_used=round(alpha, 4),
                crit_leeliu=round(lee_liu(model, pu_va.X_p, pu_va.X_u, thr), 4),
                crit_oraclef1=round(oracle_f1(model, X_va, y_va, thr), 4),
                **{k: round(v, 4) for k, v in evaluate(model, X_te, y_te, thr).items()},
                secs=round(time.time() - t0, 1))


def phase_p15(writer, fh):
    for ds, c, seed in itertools.product(DATASETS, [0.1, 0.3, 0.5, 0.7], SEEDS):
        print(f"[p15] {ds} c={c} seed={seed}", flush=True)
        pu_tr, pu_fit, Xp_hold, pu_va, val, test = prepare(ds, "scar", c, seed)
        a_bbe = bbe_alpha(pu_fit, seed)
        for lr, wd in HP_GRID:
            r = fit_and_score(pu_fit, "nnpu", a_bbe, lr, wd, seed, Xp_hold,
                              pu_va, val, test, "bbe_nnpu")
            emit(writer, fh, phase="p15", dataset=ds, mechanism="scar", c=c,
                 seed=seed, method="bbe_nnpu",
                 alpha_true=round(pu_tr.alpha_true, 4), **r)


def phase_e3(writer, fh):
    factors = [0.5, 0.75, 1.0, 1.25, 1.5]
    for ds, seed, method, k in itertools.product(DATASETS, SEEDS,
                                                 ["nnpu", "upu"], factors):
        print(f"[e3] {ds} seed={seed} {method} x{k}", flush=True)
        pu_tr, pu_fit, Xp_hold, pu_va, val, test = prepare(ds, "scar", 0.5, seed)
        alpha = min(0.999, k * pu_tr.alpha_true)
        for lr, wd in HP_GRID:
            r = fit_and_score(pu_fit, method, alpha, lr, wd, seed, Xp_hold,
                              pu_va, val, test, method)
            emit(writer, fh, phase="e3", dataset=ds, mechanism="scar", c=0.5,
                 seed=seed, method=f"{method}_x{k}",
                 alpha_true=round(pu_tr.alpha_true, 4), **r)


def phase_e4(writer, fh):
    methods = ["uasn", "wce", "en", "upu", "nnpu"]
    for ds, mech, c, seed in itertools.product(
            DATASETS, ["sar_attr", "sar_gap", "sar_extreme"], [0.3, 0.5], SEEDS):
        print(f"[e4] {ds} {mech} c={c} seed={seed}", flush=True)
        pu_tr, pu_fit, Xp_hold, pu_va, val, test = prepare(ds, mech, c, seed)
        for method in methods:
            base = method
            for lr, wd in HP_GRID:
                r = fit_and_score(pu_fit, base, pu_tr.alpha_true, lr, wd, seed,
                                  Xp_hold, pu_va, val, test, method)
                emit(writer, fh, phase="e4", dataset=ds, mechanism=mech, c=c,
                     seed=seed, method=method,
                     alpha_true=round(pu_tr.alpha_true, 4), **r)


def phase_p2(writer, fh):
    ckpts = (10, 25, 50)
    grid = [(lr, wd, h) for lr in (1e-2, 1e-3) for wd in (0.0, 1e-4)
            for h in (32, 128)]
    for ds, c, seed in itertools.product(DATASETS, [0.3, 0.5], SEEDS):
        print(f"[p2] {ds} c={c} seed={seed}", flush=True)
        pu_tr, pu_fit, Xp_hold, pu_va, val, test = prepare(ds, "scar", c, seed)
        (X_va, y_va), (X_te, y_te) = val, test
        for method in ["uasn", "wce", "en", "upu", "nnpu"]:
            for lr, wd, h in grid:
                t0 = time.time()
                snaps = train_model_ckpt(pu_fit, method, pu_tr.alpha_true,
                                         lr, wd, seed, ckpts, hidden=h)
                for ep, model in snaps.items():
                    thr = (0.5 * en_estimate_c(model, Xp_hold)
                           if method == "en" else 0.5)
                    emit(writer, fh, phase="p2", dataset=ds, mechanism="scar",
                         c=c, seed=seed, method=method, lr=lr, wd=wd, hidden=h,
                         epoch=ep, alpha_true=round(pu_tr.alpha_true, 4),
                         alpha_used=round(pu_tr.alpha_true, 4),
                         crit_leeliu=round(lee_liu(model, pu_va.X_p, pu_va.X_u, thr), 4),
                         crit_oraclef1=round(oracle_f1(model, X_va, y_va, thr), 4),
                         **{k: round(v, 4) for k, v in
                            evaluate(model, X_te, y_te, thr).items()},
                         secs=round(time.time() - t0, 1))


PHASES = {"p15": phase_p15, "e3": phase_e3, "e4": phase_e4, "p2": phase_p2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=PHASES)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2],
                    help="pass only the new seeds to extend an existing file")
    args = ap.parse_args()
    global SEEDS
    SEEDS = args.seeds
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out = os.path.join(HERE, "results", f"{args.phase}_results.csv")
    new = not os.path.exists(out)
    with open(out, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        PHASES[args.phase](writer, fh)
    print(f"{args.phase} done -> {out}")


if __name__ == "__main__":
    main()
