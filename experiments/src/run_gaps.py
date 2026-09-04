from __future__ import annotations

import argparse, csv, itertools, os, sys, time

sys.path.insert(0, os.path.dirname(__file__))
from pu_lib import (PUData, bbe_alpha, en_estimate_c, evaluate, lee_liu,
                    load_dataset, make_pu, oracle_f1, train_model_v2)
from run_ext import HP_GRID, TABULAR, prepare

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = [0, 1, 2, 3, 4]
FACTORS = [0.5, 0.75, 1.0, 1.25, 1.5]
FIELDS = ["phase", "dataset", "c", "seed", "method", "base", "factor", "lr", "wd",
          "alpha_true", "alpha_used", "crit_leeliu", "crit_oraclef1",
          "bacc", "f1", "auc", "auprc", "f1_bestthr", "thr_best",
          "precision", "recall", "fnr", "fpr", "gap", "pred_rate",
          "ece", "brier", "secs"]


def one(w, fh, phase, ds, c, seed, label, base, alpha, factor, pu_fit, Xp_hold,
        pu_va, val, test, a_true):
    X_va, y_va = val
    X_te, y_te = test
    for lr, wd in HP_GRID:
        t0 = time.time()
        model = train_model_v2(pu_fit, base, alpha, lr, wd, 50, seed, hidden=64)
        thr = 0.5 * en_estimate_c(model, Xp_hold) if base == "en" else 0.5
        row = dict(phase=phase, dataset=ds, c=c, seed=seed, method=label,
                   base=base, factor=factor, lr=lr, wd=wd,
                   alpha_true=round(a_true, 4), alpha_used=round(alpha, 4),
                   crit_leeliu=round(lee_liu(model, pu_va.X_p, pu_va.X_u, thr), 4),
                   crit_oraclef1=round(oracle_f1(model, X_va, y_va, thr), 4),
                   secs=round(time.time() - t0, 1))
        row.update({k: round(v, 5) for k, v in
                    evaluate(model, X_te, y_te, thr, extended=True).items()})
        w.writerow(row); fh.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["prior", "balance"])
    args = ap.parse_args()
    out = os.path.join(HERE, "results", f"gap_{args.phase}_results.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    new = not os.path.exists(out)
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        if args.phase == "prior":
            for ds, seed in itertools.product(TABULAR, SEEDS):
                pu_tr, pu_fit, Xp_hold, pu_va, val, test = prepare(ds, 0.5, seed)
                for base, k in itertools.product(["imbpu", "cvir", "dcpu"], FACTORS):
                    print(f"[prior] {ds} seed={seed} {base} x{k}", flush=True)
                    one(w, fh, "prior", ds, 0.5, seed, f"{base}_x{k}", base,
                        min(0.999, k * pu_tr.alpha_true), k, pu_fit, Xp_hold,
                        pu_va, val, test, pu_tr.alpha_true)
        else:
            for ds, c, seed in itertools.product(TABULAR, [0.1, 0.3, 0.5, 0.7], SEEDS):
                pu_tr, pu_fit, Xp_hold, pu_va, val, test = prepare(ds, c, seed)
                a_bbe = bbe_alpha(pu_fit, seed)
                print(f"[balance] {ds} c={c} seed={seed}", flush=True)
                for label, base, alpha in [
                        ("dcpu", "dcpu", pu_tr.alpha_true),
                        ("nnpu", "nnpu", pu_tr.alpha_true),
                        ("imbpu", "imbpu", pu_tr.alpha_true),
                        ("bbe_cvir", "cvir", a_bbe)]:
                    one(w, fh, "balance", ds, c, seed, label, base, alpha, 1.0,
                        pu_fit, Xp_hold, pu_va, val, test, pu_tr.alpha_true)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
