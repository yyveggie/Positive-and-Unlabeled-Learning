from __future__ import annotations

import argparse, csv, itertools, os, sys, time, warnings
import numpy as np
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(HERE, "external", "sarpu", "sarpu"))

from pu_lib import driving_feature, load_dataset, make_pu, metrics_from_scores
from sarpu.pu_learning import pu_learn_neg, pu_learn_sar_em, pu_learn_scar_c

warnings.filterwarnings("ignore")

DATASETS = ["breast", "spambase", "mushroom", "adult", "bank"]
MECHANISMS = ["scar", "sar_indep", "sar_attr", "sar_gap", "sar_extreme"]
CS = [0.3, 0.5]
FIELDS = ["dataset", "mechanism", "c", "seed", "method", "prop_spec",
          "alpha_true", "bacc", "f1", "auc", "auprc", "f1_bestthr", "thr_best",
          "precision", "recall", "pred_rate", "ece", "brier", "iters", "secs"]


def propensity_mask(mechanism, X, y, spec):
    d = X.shape[1]
    if spec == "all":
        return np.arange(d)
    if mechanism == "scar":
        return np.array([], dtype=int)
    if mechanism == "sar_gap":
        return np.arange(d)
    mode = "min" if mechanism == "sar_indep" else "max"
    return np.array([driving_feature(X, y, mode)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=DATASETS,
                    help="one per shard when running in parallel; EM on the "
                         "larger sets dominates the wall clock")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2],
                    help="pass only the new seeds to extend an existing file")
    ap.add_argument(
        "--failure-log", default=None,
        help="separate row-level failure CSV (default: sarem_failures.csv beside --out)",
    )
    args = ap.parse_args()
    out = args.out or os.path.join(HERE, "results", "sarem_results.csv")
    failure_out = args.failure_log or out.replace(".csv", "_failures.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    failure_fields = ["dataset", "mechanism", "c", "seed", "method",
                      "prop_spec", "status", "error"]
    failure_new = not os.path.exists(failure_out)
    new = not os.path.exists(out)
    with open(out, "a", newline="") as fh, open(failure_out, "a", newline="") as ffh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        fw = csv.DictWriter(ffh, fieldnames=failure_fields)
        if new:
            w.writeheader()
        if failure_new:
            fw.writeheader()
        for ds, mech, c, seed in itertools.product(args.datasets, MECHANISMS,
                                                   CS, args.seeds):
            print(f"[sarem] {ds} {mech} c={c} seed={seed}", flush=True)
            X, y = load_dataset(ds, seed=seed)
            X_tr, X_tmp, y_tr, y_tmp = train_test_split(
                X, y, test_size=0.4, stratify=y, random_state=seed)
            _, X_te, _, y_te = train_test_split(
                X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed)
            mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-9
            X_tr, X_te = (X_tr - mu) / sd, (X_te - mu) / sd
            pu = make_pu(X_tr, y_tr, c, mech, seed)
            s = pu.s.astype(float)
            c_true = float(s.sum() / (y_tr.sum() + 1e-9))

            runs = [("neg", "-", lambda: pu_learn_neg(X_tr, s)),
                    ("scar_c", "-", lambda: pu_learn_scar_c(X_tr, s, c_true))]
            for spec in ("oracle", "all"):
                mask = propensity_mask(mech, X_tr, y_tr, spec)
                runs.append(("sar_em", spec,
                             lambda m=mask: pu_learn_sar_em(
                                 X_tr, s, propensity_attributes=m)))

            for name, spec, fn in runs:
                try:
                    t0 = time.time()
                    res = fn()
                    model, info = res[0], res[-1]
                    p = np.asarray(model.predict_proba(X_te)).ravel()
                    w.writerow(dict(
                        dataset=ds, mechanism=mech, c=c, seed=seed, method=name,
                        prop_spec=spec, alpha_true=round(pu.alpha_true, 4),
                        **{k: round(v, 4) for k, v in
                           metrics_from_scores(p, y_te, 0.5,
                                               extended=True).items()},
                        iters=info.get("nb_iterations", ""),
                        secs=round(time.time() - t0, 1)))
                    fh.flush()
                except Exception as e:
                    print(f"  FAILED {name}/{spec}: {type(e).__name__}: {e}",
                          flush=True)
                    fw.writerow(dict(
                        dataset=ds, mechanism=mech, c=c, seed=seed, method=name,
                        prop_spec=spec, status="failed",
                        error=f"{type(e).__name__}: {e}",
                    ))
                    ffh.flush()
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
