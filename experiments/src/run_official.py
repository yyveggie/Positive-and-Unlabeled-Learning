from __future__ import annotations

import argparse, csv, itertools, os, sys, time
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "methods"))
from pu_lib import PUData, load_dataset, make_pu
from official_adapters import train_distpu_official, train_vpu_official

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    return (make_pu(X_tr, y_tr, c, "scar", seed),
            make_pu(X_va, y_va, c, "scar", seed + 1000),
            (X_va, y_va), (X_te, y_te))


def run_cell(writer, fh, phase, ds, c, seed, hidden=64, max_rows=8000,
             warm_epochs=60, pu_epochs=60):
    pu_tr, pu_va, (X_va, y_va), (X_te, y_te) = prepare(ds, c, seed, max_rows)
    for method in ["vpu_off", "distpu_off"]:
        t0 = time.time()
        if method == "vpu_off":
            pred = train_vpu_official(pu_tr, pu_va, seed)
            alpha_used = float("nan")
        else:
            pred = train_distpu_official(pu_tr, pu_tr.alpha_true, seed,
                                         warm_epochs=warm_epochs,
                                         pu_epochs=pu_epochs, hidden=hidden)
            alpha_used = pu_tr.alpha_true
        p_te = pred.proba(X_te)
        yhat = p_te > 0.5
        r = float((pred.proba(pu_va.X_p) > 0.5).mean())
        pr1 = float((pred.proba(pu_va.X_u) > 0.5).mean())
        writer.writerow(dict(
            phase=phase, dataset=ds, mechanism="scar", c=c, seed=seed,
            method=method, lr="", wd="", hidden=hidden,
            epoch=warm_epochs + pu_epochs,
            alpha_true=round(pu_tr.alpha_true, 4),
            alpha_used=round(alpha_used, 4) if alpha_used == alpha_used else "",
            crit_leeliu=round(r * r / max(pr1, 1e-6), 4),
            crit_oraclef1=round(f1_score(y_va, pred.proba(X_va) > 0.5,
                                         zero_division=0), 4),
            bacc=round(balanced_accuracy_score(y_te, yhat), 4),
            f1=round(f1_score(y_te, yhat, zero_division=0), 4),
            auc=round(roc_auc_score(y_te, p_te), 4),
            secs=round(time.time() - t0, 1)))
        fh.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["smoke", "tab", "mnist", "fix"])
    args = ap.parse_args()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out = os.path.join(HERE, "results", "official_results.csv")
    new = not os.path.exists(out)
    with open(out, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        if args.phase == "smoke":
            run_cell(writer, fh, "smoke", "breast", 0.5, 0,
                     warm_epochs=20, pu_epochs=20)
        elif args.phase == "tab":
            for ds, c, seed in itertools.product(
                    ["breast", "spambase", "mushroom", "adult", "bank"],
                    [0.1, 0.3, 0.5, 0.7], [0, 1, 2]):
                print(f"[tab] {ds} c={c} seed={seed}", flush=True)
                try:
                    run_cell(writer, fh, "tab", ds, c, seed)
                except Exception as e:
                    print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
        elif args.phase == "fix":
            cells = ([("adult", 0.7, s) for s in (0, 1, 2)]
                     + [("bank", c, s) for c in (0.1, 0.3, 0.5, 0.7)
                        for s in (0, 1, 2)])
            for ds, c, seed in cells:
                print(f"[fix] {ds} c={c} seed={seed}", flush=True)
                try:
                    run_cell(writer, fh, "tab", ds, c, seed)
                except Exception as e:
                    print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
        else:
            for c, seed in itertools.product([0.3, 0.5], [0, 1, 2]):
                print(f"[mnist] c={c} seed={seed}", flush=True)
                try:
                    run_cell(writer, fh, "mnist", "mnist20k", c, seed,
                             hidden=128, max_rows=20000)
                except Exception as e:
                    print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
