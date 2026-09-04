from __future__ import annotations

import argparse, csv, itertools, os, sys, time

sys.path.insert(0, os.path.dirname(__file__))
from pu_lib import (PUData, bbe_alpha, criteria_from_scores, en_estimate_c,
                    load_dataset, make_pu, metrics_from_scores, oracle_f1,
                    scores, train_model_ckpt)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = ["breast", "spambase", "mushroom", "adult", "bank"]
METHODS = ["uasn", "wce", "en", "upu", "nnpu", "imbpu"]
GRID = [(lr, wd, h) for lr in (1e-2, 1e-3) for wd in (0.0, 1e-4)
        for h in (32, 128)]
CKPTS = (10, 25, 50)
FIELDS = ["dataset", "c", "seed", "method", "lr", "wd", "hidden", "epoch",
          "alpha_true", "alpha_hat",
          "crit_leeliu", "crit_puauc", "crit_nnpurisk", "crit_f1hat",
          "crit_nnpurisk_true", "crit_f1hat_true", "crit_oraclef1",
          "bacc", "f1", "auc", "secs"]


def prepare(ds, c, seed):
    from sklearn.model_selection import train_test_split
    X, y = load_dataset(ds, seed=seed)
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


def run_cell(w, fh, ds, c, seed):
    pu_tr, pu_fit, Xp_hold, pu_va, (X_va, y_va), (X_te, y_te) = \
        prepare(ds, c, seed)
    a_true = pu_tr.alpha_true
    a_hat = bbe_alpha(pu_fit, seed)
    for method in METHODS:
        for lr, wd, h in GRID:
            t0 = time.time()
            snaps = train_model_ckpt(pu_fit, method, a_true, lr, wd, seed,
                                     CKPTS, hidden=h)
            for ep, m in snaps.items():
                thr = 0.5 * en_estimate_c(m, Xp_hold) if method == "en" else 0.5
                sp, su = scores(m, pu_va.X_p), scores(m, pu_va.X_u)
                est = criteria_from_scores(sp, su, a_hat, thr)
                tru = criteria_from_scores(sp, su, a_true, thr)
                w.writerow(dict(
                    dataset=ds, c=c, seed=seed, method=method, lr=lr, wd=wd,
                    hidden=h, epoch=ep, alpha_true=round(a_true, 4),
                    alpha_hat=round(a_hat, 4),
                    crit_leeliu=round(est["crit_leeliu"], 4),
                    crit_puauc=round(est["crit_puauc"], 4),
                    crit_nnpurisk=round(est["crit_nnpurisk"], 4),
                    crit_f1hat=round(est["crit_f1hat"], 4),
                    crit_nnpurisk_true=round(tru["crit_nnpurisk"], 4),
                    crit_f1hat_true=round(tru["crit_f1hat"], 4),
                    crit_oraclef1=round(oracle_f1(m, X_va, y_va, thr), 4),
                    **{k: round(v, 4) for k, v in
                       metrics_from_scores(scores(m, X_te), y_te, thr).items()},
                    secs=round(time.time() - t0, 1)))
            fh.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2],
                    help="pass only the new seeds to extend an existing file")
    args = ap.parse_args()
    out = args.out or os.path.join(HERE, "results", "crit_results.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cells = ([("breast", 0.5, 0)] if args.smoke else
             list(itertools.product(DATASETS, [0.3, 0.5], args.seeds)))
    new = not os.path.exists(out)
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for ds, c, seed in cells:
            print(f"[crit] {ds} c={c} seed={seed}", flush=True)
            try:
                run_cell(w, fh, ds, c, seed)
            except Exception as e:
                print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
