from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
KEY = ["dataset", "alpha_target", "seed", "method"]
DATASETS = {"spambase", "adult", "bank", "mushroom"}
METHODS = {"uasn", "wce", "nnpu", "imbpu", "cvir", "bbe_cvir"}
PRIORS = [0.10, 0.075, 0.05, 0.03, 0.02, 0.01]
SEEDS = {0, 1, 2, 3, 4}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pattern", default=str(ROOT / "results" / "rare_dense_seed*.csv")
    )
    ap.add_argument(
        "--out", default=str(ROOT / "results" / "rare_dense_track.csv")
    )
    args = ap.parse_args()

    pattern = Path(args.pattern)
    files = sorted(pattern.parent.glob(pattern.name))
    if not files:
        raise FileNotFoundError(args.pattern)
    data = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if data.duplicated(KEY).any():
        raise RuntimeError("duplicate rare-grid design keys")
    failed = data[data.status != "ok"]
    if len(failed):
        raise RuntimeError(f"{len(failed)} failed rare-grid runs")
    expected = len(DATASETS) * len(METHODS) * len(PRIORS) * 5
    if len(data) != expected:
        raise RuntimeError(f"incomplete rare grid: {len(data)} != {expected}")
    if set(data.dataset) != DATASETS or set(data.method) != METHODS:
        raise RuntimeError("unexpected dataset or method levels")
    if not np.allclose(sorted(data.alpha_target.unique()), sorted(PRIORS)):
        raise RuntimeError("unexpected prior levels")
    if set(data.seed) != SEEDS or set(data.c) != {0.5}:
        raise RuntimeError("unexpected rare-grid seed or label-frequency levels")
    if not (data.groupby(["dataset", "alpha_target", "method"]).size() == 5).all():
        raise RuntimeError("unbalanced rare-grid factor cells")
    max_u_deviation = float((data.alpha_unlabeled - data.alpha_target).abs().max())
    max_test_deviation = float((data.alpha_test - data.alpha_target).abs().max())
    if max_u_deviation > 3e-5 or max_test_deviation > 7e-4:
        raise RuntimeError(
            "realized prior exceeds declared integer-rounding tolerance: "
            f"U={max_u_deviation:g}, test={max_test_deviation:g}"
        )
    if set(data.epochs) != {50}:
        raise RuntimeError(f"rare-grid epoch budget changed: {set(data.epochs)}")
    data = data.sort_values(KEY).reset_index(drop=True)
    out = Path(args.out)
    data.to_csv(out, index=False)

    data["degenerate"] = (
        (data.pred_rate <= 1e-8) | (data.pred_rate >= 1 - 1e-8)
    )
    data["near_chance_ap"] = data.auprc <= 2.0 * data.alpha_test
    metrics = ["f1", "auprc", "normalized_ap_lift", "f1_bestthr", "auc"]
    summary = data.groupby(["alpha_target", "method"], as_index=False).agg(
        **{f"mean_{m}": (m, "mean") for m in metrics},
        **{f"sd_{m}": (m, "std") for m in metrics},
        degenerate_rate=("degenerate", "mean"),
        near_chance_ap_rate=("near_chance_ap", "mean"),
        n=("seed", "size"),
    )
    summary.to_csv(out.with_name("rare_dense_summary.csv"), index=False)


    interval_rows = []
    rng = np.random.default_rng(20260815)

    def dataset_bootstrap(part, column_high, column_low, n_boot=10000):
        tmp = part.assign(drop=part[column_high] - part[column_low])
        means = tmp.groupby("dataset")["drop"].mean().to_numpy()
        sampled = rng.integers(0, len(means), size=(n_boot, len(means)))
        return np.quantile(means[sampled].mean(axis=1), [0.025, 0.975])

    for high, low in zip(PRIORS[:-1], PRIORS[1:]):
        left = data[data.alpha_target == high]
        right = data[data.alpha_target == low]
        pair = left.merge(
            right, on=["dataset", "seed", "method"], suffixes=("_high", "_low"),
            validate="one_to_one",
        )
        for method, part in pair.groupby("method"):
            drop = part.normalized_ap_lift_high - part.normalized_ap_lift_low
            lo_ci, hi_ci = dataset_bootstrap(
                part, "normalized_ap_lift_high", "normalized_ap_lift_low"
            )
            interval_rows.append(dict(
                method=method, high_prior=high, low_prior=low,
                interval=f"{100*high:g}->{100*low:g}%",
                mean_drop=drop.mean(), sd_drop=drop.std(ddof=1),
                ci95_low=lo_ci, ci95_high=hi_ci, n_pairs=len(drop),
            ))
        drop = pair.normalized_ap_lift_high - pair.normalized_ap_lift_low
        lo_ci, hi_ci = dataset_bootstrap(
            pair, "normalized_ap_lift_high", "normalized_ap_lift_low"
        )
        interval_rows.append(dict(
            method="pooled", high_prior=high, low_prior=low,
            interval=f"{100*high:g}->{100*low:g}%",
            mean_drop=drop.mean(), sd_drop=drop.std(ddof=1),
            ci95_low=lo_ci, ci95_high=hi_ci, n_pairs=len(drop),
        ))
    intervals = pd.DataFrame(interval_rows)
    intervals.to_csv(out.with_name("rare_dense_adjacent_drops.csv"), index=False)


    curve = data.groupby(
        ["dataset", "method", "alpha_target"], as_index=False
    ).normalized_ap_lift.mean()
    steepest = []
    for (dataset, method), part in curve.groupby(["dataset", "method"]):
        values = part.set_index("alpha_target").normalized_ap_lift.reindex(PRIORS)
        drops = values.to_numpy()[:-1] - values.to_numpy()[1:]
        index = int(np.nanargmax(drops))
        steepest.append(dict(
            dataset=dataset, method=method,
            high_prior=PRIORS[index], low_prior=PRIORS[index + 1],
            interval=f"{100*PRIORS[index]:g}->{100*PRIORS[index+1]:g}%",
            normalized_ap_drop=drops[index],
        ))
    steepest = pd.DataFrame(steepest)
    steepest.to_csv(out.with_name("rare_dense_steepest_intervals.csv"), index=False)
    steepest.groupby("interval").size().rename("curve_count").reset_index().to_csv(
        out.with_name("rare_dense_steepest_counts.csv"), index=False
    )

    bbe = data[data.method == "bbe_cvir"].groupby(
        "alpha_target", as_index=False
    ).agg(
        mean_alpha_hat=("alpha_used", "mean"),
        mean_ratio=("alpha_estimation_ratio", "mean"),
        median_ratio=("alpha_estimation_ratio", "median"),
        mean_auprc=("auprc", "mean"),
    )
    bbe.to_csv(out.with_name("rare_dense_bbe_summary.csv"), index=False)
    counts = steepest.groupby("interval").size().sort_values(ascending=False)
    print(f"wrote {out} ({len(data)} complete rows); steepest intervals:\n{counts}")


if __name__ == "__main__":
    main()
