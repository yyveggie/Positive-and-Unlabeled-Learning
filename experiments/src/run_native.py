from __future__ import annotations

import argparse, csv, os, sys, time

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "methods"))
from pu_lib import (PUData, bbe_alpha, criteria_from_scores, en_estimate_c,
                    lee_liu, metrics_from_scores, oracle_f1, scores,
                    train_model_v2)
from official_adapters import OFFICIAL_GRID, train_official
from scar_test import scar_test

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "features", "uniprot_nucleus.npz")
METHODS = ["uasn", "wce", "en", "upu", "nnpu", "imbpu", "cvir",
           "bbe_nnpu", "bbe_cvir"]
OFFICIAL = ["vpu_off", "distpu_off"]
HP = [(lr, wd) for lr in (1e-2, 1e-3) for wd in (0.0, 1e-4)]
FACTORS = [0.5, 0.75, 1.0, 1.25, 1.5]
EPOCHS = 50
FIELDS = ["phase", "seed", "method", "lr", "wd", "cfg", "alpha_true",
          "alpha_used", "c_true", "crit_leeliu", "crit_oraclef1",
          "bacc", "f1", "auc", "auprc", "f1_bestthr", "thr_best",
          "precision", "recall", "pred_rate", "ece", "brier", "secs"]


def load():
    d = np.load(DATA, allow_pickle=True)
    return d["X"], d["y"].astype(int), d["s"].astype(int)


def split(X, y, s, seed):
    idx = np.arange(len(y))
    i_tr, i_tmp = train_test_split(idx, test_size=0.4, stratify=y,
                                   random_state=seed)
    i_va, i_te = train_test_split(i_tmp, test_size=0.5, stratify=y[i_tmp],
                                  random_state=seed)
    mu, sd = X[i_tr].mean(0), X[i_tr].std(0) + 1e-9
    Z = (X - mu) / sd

    def pu(ix):
        return PUData(X_p=Z[ix][s[ix] == 1], X_u=Z[ix], y_u=y[ix],
                      alpha_true=float(y[ix].mean()),
                      c=float(s[ix].sum() / max(y[ix].sum(), 1)),
                      s=s[ix])
    return pu(i_tr), pu(i_va), (Z[i_va], y[i_va]), (Z[i_te], y[i_te])


def run_seed(w, fh, seed, smoke=False, methods=METHODS, official=OFFICIAL,
             do_prior=True):
    X, y, s = load()
    pu_tr, pu_va, (X_va, y_va), (X_te, y_te) = split(X, y, s, seed)
    n_hold = max(1, len(pu_tr.X_p) // 5)
    pu_fit = PUData(pu_tr.X_p[:-n_hold], pu_tr.X_u, pu_tr.y_u,
                    pu_tr.alpha_true, pu_tr.c)
    hold = pu_tr.X_p[-n_hold:]
    a_true, a_bbe = pu_tr.alpha_true, bbe_alpha(pu_fit, seed)
    print(f"  seed {seed}: alpha {a_true:.4f}, c {pu_tr.c:.4f}, "
          f"BBE alpha-hat {a_bbe:.4f} (ratio {a_bbe / a_true:.2f})", flush=True)
    common = dict(seed=seed, alpha_true=round(a_true, 5),
                  c_true=round(pu_tr.c, 5))

    def emit(phase, method, alpha, model_scores, crit_leeliu, crit_or, secs,
             lr="", wd="", cfg="", thr=0.5):
        w.writerow(dict(
            common, phase=phase, method=method, lr=lr, wd=wd, cfg=cfg,
            alpha_used="" if alpha != alpha else round(alpha, 5),
            crit_leeliu=round(crit_leeliu, 4), crit_oraclef1=round(crit_or, 4),
            **{k: round(v, 4) for k, v in
               metrics_from_scores(model_scores, y_te, thr, extended=True).items()},
            secs=round(secs, 1)))
        fh.flush()

    for method in (methods[:2] if smoke else methods):
        base = {"bbe_nnpu": "nnpu", "bbe_cvir": "cvir"}.get(method, method)
        alpha = a_bbe if method.startswith("bbe_") else a_true
        for lr, wd in HP:
            t0 = time.time()
            m = train_model_v2(pu_fit, base, alpha, lr, wd, EPOCHS, seed)
            thr = 0.5 * en_estimate_c(m, hold) if method == "en" else 0.5
            emit("main", method, alpha, scores(m, X_te),
                 lee_liu(m, pu_va.X_p, pu_va.X_u, thr),
                 oracle_f1(m, X_va, y_va, thr), time.time() - t0, lr, wd,
                 thr=thr)

    for method in ([] if smoke else official):
        for cfg in OFFICIAL_GRID[method]:
            t0 = time.time()
            try:
                pred = train_official(method, pu_fit, pu_va, a_true, seed, cfg)
                crit = criteria_from_scores(pred.proba(pu_va.X_p),
                                            pred.proba(pu_va.X_u))
                emit("main", method,
                     float("nan") if method == "vpu_off" else a_true,
                     pred.proba(X_te), crit["crit_leeliu"],
                     f1_score(y_va, pred.proba(X_va) > 0.5, zero_division=0),
                     time.time() - t0,
                     cfg="|".join(f"{k}{v}" for k, v in cfg.items()))
            except Exception as e:
                print(f"    FAILED {method} {cfg}: {type(e).__name__}: {e}",
                      flush=True)


    for k in ([] if smoke or not do_prior else FACTORS):
        for lr, wd in HP:
            t0 = time.time()
            a = min(0.999, k * a_true)
            m = train_model_v2(pu_fit, "nnpu", a, lr, wd, EPOCHS, seed)
            emit("prior", f"nnpu_x{k}", a, scores(m, X_te),
                 lee_liu(m, pu_va.X_p, pu_va.X_u), oracle_f1(m, X_va, y_va),
                 time.time() - t0, lr, wd)


def oracle_scar_check(Z, y, s, B=200, seed=0):
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    lab = s[pos].astype(bool)
    from scar_test import STATS
    f = STATS["ks"]
    T0 = f(Z[pos][lab], Z[pos][~lab])
    nulls = np.empty(B)
    for b in range(B):
        perm = rng.permutation(len(pos))
        m = lab[perm]
        nulls[b] = f(Z[pos][m], Z[pos][~m])
    return {"p": float(np.mean(nulls >= T0)), "T0": float(T0),
            "null_mean": float(nulls.mean()), "n_labeled": int(lab.sum()),
            "n_hidden": int((~lab).sum())}


def run_scar_test(out_path, seeds, B=200):
    X, y, s = load()
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    rows = []
    for stat in ["ks", "nb_auc", "kl"]:
        for seed in seeds:
            r = scar_test(Z, s, prior=float(y.mean()), stat=stat, B=B,
                          seed=seed, y_true=y)
            r.update(stat=stat, seed=seed, prior=round(float(y.mean()), 4))
            rows.append(r)
            print(f"  SCAR test [{stat}] seed {seed}: p = {r['p']:.3f}, "
                  f"reject = {r['reject']}, purity of P-hat = "
                  f"{r.get('purity', float('nan')):.3f}", flush=True)
    orc = oracle_scar_check(Z, y, s, B=B, seed=0)
    print(f"  ground-truth check (labeled positives vs hidden positives): "
          f"KS = {orc['T0']:.3f} against a null mean of {orc['null_mean']:.3f}, "
          f"p = {orc['p']:.3f}", flush=True)
    rows.append(dict(orc, stat="oracle_permutation", seed=0,
                     prior=round(float(y.mean()), 4),
                     reject=int(orc["p"] < 0.05)))
    keys = sorted({k for r in rows for k in r})
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2],
                    help="pass only the new seeds to extend an existing file")
    ap.add_argument("--methods", nargs="*", default=METHODS)
    ap.add_argument("--official", nargs="*", default=OFFICIAL)
    ap.add_argument("--no-prior-sweep", action="store_true")
    ap.add_argument("--no-scartest", action="store_true")
    args = ap.parse_args()
    out = args.out or os.path.join(HERE, "results", "native_results.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    seeds = [0] if args.smoke else args.seeds
    new = not os.path.exists(out)
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for seed in seeds:
            print(f"[native] seed {seed}", flush=True)
            run_seed(w, fh, seed, args.smoke, args.methods, args.official,
                     not args.no_prior_sweep)
    if args.no_scartest:
        print(f"done -> {out}"); return
    print("[native] SCAR test on the real labeling", flush=True)
    run_scar_test(out.replace(".csv", "_scartest.csv"), seeds,
                  B=20 if args.smoke else 200)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
