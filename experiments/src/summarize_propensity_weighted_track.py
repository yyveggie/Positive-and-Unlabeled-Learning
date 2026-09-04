from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
KEY = ["dataset", "mechanism", "c", "seed"]
DATASETS = {"breast", "spambase", "mushroom", "adult", "bank"}
MECHANISMS = {"scar", "sar_indep", "sar_attr", "sar_gap", "sar_extreme"}
CS = {0.3, 0.5}
SEEDS = {0, 1, 2, 3, 4}


def cluster_bootstrap(values: pd.DataFrame, rng, n_boot=10000):
    by_dataset = values.groupby("dataset")["difference"].mean().to_numpy()
    sampled = rng.integers(0, len(by_dataset), size=(n_boot, len(by_dataset)))
    draws = by_dataset[sampled].mean(axis=1)
    return np.quantile(draws, [0.025, 0.975])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pattern",
        default=str(ROOT / "results" / "propensity_weighted_seed*.csv"),
    )
    ap.add_argument(
        "--sarem", default=str(ROOT / "results" / "sarem_results.csv")
    )
    ap.add_argument(
        "--out", default=str(ROOT / "results" / "propensity_weighted_track.csv")
    )
    args = ap.parse_args()

    pattern = Path(args.pattern)
    files = sorted(pattern.parent.glob(pattern.name))
    if not files:
        raise FileNotFoundError(args.pattern)
    pg = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if pg.duplicated(KEY).any():
        raise RuntimeError("duplicate PG-IPW design keys")
    failed = pg[pg.status != "ok"]
    if len(failed):
        raise RuntimeError(f"{len(failed)} PG-IPW failures:\n{failed[KEY + ['error']]}")
    expected = len(DATASETS) * len(MECHANISMS) * 2 * 5
    if len(pg) != expected:
        raise RuntimeError(f"PG-IPW grid incomplete: {len(pg)} != {expected}")
    if set(pg.dataset) != DATASETS or set(pg.mechanism) != MECHANISMS:
        raise RuntimeError("unexpected PG-IPW factor levels")
    if set(pg.c) != CS or set(pg.seed) != SEEDS:
        raise RuntimeError("unexpected PG-IPW label-frequency or seed levels")
    if not (pg.groupby(["dataset", "mechanism", "c"]).size() == 5).all():
        raise RuntimeError("unbalanced PG-IPW factor cells")
    pg = pg.sort_values(KEY).reset_index(drop=True)
    out = Path(args.out)
    pg.to_csv(out, index=False)

    old = pd.read_csv(args.sarem)
    old["method_id"] = np.where(
        old.method == "sar_em", "sar_em_" + old.prop_spec.astype(str), old.method
    )
    old["status"] = "ok"
    pg_combined = pg.copy()
    pg_combined["method_id"] = "pg_ipw"
    pg_combined["prop_spec"] = pg_combined["posterior_model"]
    combined = pd.concat([old, pg_combined], ignore_index=True, sort=False)
    combined.to_csv(out.with_name("beyond_scar_combined.csv"), index=False)

    metrics = ["bacc", "f1", "auc", "auprc", "f1_bestthr", "ece", "brier"]
    summary = combined.groupby(["mechanism", "method_id"], as_index=False).agg(
        **{f"mean_{m}": (m, "mean") for m in metrics},
        **{f"sd_{m}": (m, "std") for m in metrics},
        n=("seed", "size"),
    )
    summary.to_csv(out.with_name("beyond_scar_summary.csv"), index=False)

    rng = np.random.default_rng(20260815)
    paired_rows = []
    refs = ("scar_c", "sar_em_all", "sar_em_oracle")
    for reference in refs:
        ref = combined[combined.method_id == reference]
        cand = combined[combined.method_id == "pg_ipw"]
        pair = cand.merge(ref, on=KEY, suffixes=("_pg", "_ref"), validate="one_to_one")
        for mechanism in sorted(MECHANISMS):
            part = pair[pair.mechanism == mechanism]
            for metric in metrics:
                values = pd.DataFrame({
                    "dataset": part.dataset,
                    "difference": part[f"{metric}_pg"] - part[f"{metric}_ref"],
                }).dropna()
                lo, hi = cluster_bootstrap(values, rng)
                paired_rows.append(dict(
                    reference=reference, mechanism=mechanism, metric=metric,
                    difference_definition="PG-IPW minus reference",
                    mean_difference=values.difference.mean(),
                    ci95_low=lo, ci95_high=hi, n_pairs=len(values),
                    n_datasets=values.dataset.nunique(),
                ))
    pd.DataFrame(paired_rows).to_csv(
        out.with_name("beyond_scar_paired_contrasts.csv"), index=False
    )
    print(
        f"wrote {out} ({len(pg)} PG-IPW rows), combined={len(combined)}; "
        f"SAR-EM has {1000 - len(old)} recorded numerical failures"
    )


if __name__ == "__main__":
    main()
