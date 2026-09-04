from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from run_temporal_native import prospective_metrics


HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"
PREDICTIONS = RESULTS / "temporal_native_predictions"
METHOD_ORDER = [
    "uasn", "wce", "upu_bbe", "nnpu_bbe", "imbpu_bbe", "cvir_bbe",
    "vpu_off", "distpu_bbe_off", "holistic_off",
]
METRICS = ["auc", "ap", "recall_at_1pct", "enrichment_at_1pct",
           "recall_at_5pct", "enrichment_at_5pct"]


def load_results(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    data = pd.concat(frames, ignore_index=True)
    keys = ["dataset", "seed", "method", "lr", "wd", "cfg"]
    duplicate = data.duplicated(keys, keep=False)
    if duplicate.any():


        nonkeys = [column for column in data.columns if column not in keys]
        for _, group in data[duplicate].groupby(keys, dropna=False):
            if any(group[column].nunique(dropna=False) > 1 for column in nonkeys):
                raise RuntimeError("non-identical duplicate configuration rows found")
        data = data.drop_duplicates(keys, keep="last")
    expected = set(METHOD_ORDER)
    if set(data.method) != expected:
        raise RuntimeError(f"method roster mismatch: {sorted(set(data.method) ^ expected)}")
    selected = data[data.selected == 1]
    counts = selected.groupby(["dataset", "seed", "method"]).size()
    if not (counts == 1).all():
        raise RuntimeError("each dataset/seed/method must select exactly one configuration")
    metric_columns = ["auc", "ap", "recall_at_1pct", "precision_at_1pct",
                      "enrichment_at_1pct", "recall_at_5pct",
                      "precision_at_5pct", "enrichment_at_5pct"]
    if data.loc[data.selected == 0, metric_columns].notna().any().any():
        raise RuntimeError("a nonselected configuration contains test metrics")
    if selected[metric_columns].isna().any().any():
        raise RuntimeError("a selected configuration is missing test metrics")
    return data


def load_prediction_cube(dataset: str, methods: list[str], seeds: list[int]):
    cube, entities, event = {}, None, None
    for method in methods:
        per_seed = []
        for seed in seeds:
            path = PREDICTIONS / f"{dataset}_seed{seed}_{method}.npz"
            d = np.load(path, allow_pickle=True)
            ids = d["entity_ids"].astype(str)
            y = d["event"].astype(np.int8)
            if entities is None:
                entities, event = ids, y
            elif not np.array_equal(ids, entities) or not np.array_equal(y, event):
                raise RuntimeError(f"prediction alignment mismatch at {path}")
            per_seed.append(d["score"].astype(np.float64))
        cube[method] = np.stack(per_seed)
    return entities, event, cube


def load_debian_audit_cube(methods: list[str], seeds: list[int]):
    cohort = np.load(HERE / "features" / "native_temporal" /
                     "debian_vulnerability_temporal.npz", allow_pickle=True)
    ids = cohort["packages"].astype(str)
    labels = cohort["labels"].astype(bool)
    present = cohort["present"].astype(bool)
    known_by = cohort["known_by"].astype(bool)
    mask = present[:, 3] & (~known_by[:, 2])
    event = labels[mask, 3].astype(np.int8)
    cube = {}
    for method in methods:
        per_seed = []
        for seed in seeds:
            path = PREDICTIONS / f"debian_seed{seed}_{method}.npz"
            d = np.load(path, allow_pickle=True)
            prediction_ids = d["cohort_entity_ids"].astype(str)
            if not np.array_equal(prediction_ids, ids):
                raise RuntimeError(f"audit prediction alignment mismatch at {path}")
            per_seed.append(d["cohort_score"].astype(np.float64)[mask])
        cube[method] = np.stack(per_seed)
    return ids[mask], event, cube


def fast_prospective_metrics(scores: np.ndarray, event: np.ndarray) -> dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64)
    event = np.asarray(event, dtype=bool)
    n, positives = len(event), int(event.sum())
    negatives = n - positives
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores, sorted_event = scores[order], event[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores) != 0) + 1]
    sizes = np.diff(np.r_[starts, n])
    positive_group = np.add.reduceat(sorted_event.astype(np.int64), starts)
    negative_group = sizes - positive_group
    cumulative_positive = np.cumsum(positive_group)
    cumulative_negative = np.cumsum(negative_group)
    precision_at_group = cumulative_positive / (cumulative_positive + cumulative_negative)
    ap = np.sum((positive_group / positives) * precision_at_group)
    negatives_below = negatives - cumulative_negative
    auc = np.sum(positive_group * (negatives_below + 0.5 * negative_group)) \
        / (positives * negatives)
    rate = positives / n
    out = {"event_rate": rate, "auc": float(auc), "ap": float(ap)}
    for fraction, label in ((0.01, "1pct"), (0.05, "5pct")):
        k = max(1, int(np.ceil(fraction * n)))
        hits = int(sorted_event[:k].sum())
        precision = hits / k
        out[f"recall_at_{label}"] = hits / positives
        out[f"precision_at_{label}"] = precision
        out[f"enrichment_at_{label}"] = precision / rate
    return out


def paired_hierarchical_bootstrap(dataset: str, methods: list[str], seeds: list[int],
                                  repetitions: int, rng: np.random.Generator):
    if dataset == "debian_audit_2026":
        _, event, cube = load_debian_audit_cube(methods, seeds)
    else:
        _, event, cube = load_prediction_cube(dataset, methods, seeds)
    n, n_seeds = len(event), len(seeds)
    values = {metric: {method: np.empty(repetitions) for method in methods}
              for metric in METRICS}
    for b in range(repetitions):
        entity_index = rng.integers(0, n, n)
        seed_index = rng.integers(0, n_seeds, n_seeds)
        y = event[entity_index]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        for method in methods:


            per_seed = [fast_prospective_metrics(cube[method][s, entity_index], y)
                        for s in seed_index]
            for metric in METRICS:
                values[metric][method][b] = np.mean([row[metric] for row in per_seed])
    return values


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)
    m, adjusted, running = len(ordered), {}, 0.0
    for rank, key in enumerate(ordered):
        value = min(1.0, (m - rank) * pvalues[key])
        running = max(running, value)
        adjusted[key] = running
    return adjusted


def add_debian_audit(cube_rows: list[dict], methods: list[str], seeds: list[int]):
    cohort = np.load(HERE / "features" / "native_temporal" /
                     "debian_vulnerability_temporal.npz", allow_pickle=True)
    ids = cohort["packages"].astype(str)
    labels = cohort["labels"].astype(bool)
    present = cohort["present"].astype(bool)
    known_by = cohort["known_by"].astype(bool)
    audit = labels[:, 3]
    mask = present[:, 3] & (~known_by[:, 2])
    wanted_ids, wanted_event = ids[mask], audit[mask]
    for method in methods:
        for seed in seeds:
            d = np.load(PREDICTIONS / f"debian_seed{seed}_{method}.npz", allow_pickle=True)
            prediction_ids = d["cohort_entity_ids"].astype(str)
            lookup = {entity: i for i, entity in enumerate(prediction_ids)}
            positions = np.array([lookup[entity] for entity in wanted_ids])
            scores = d["cohort_score"].astype(np.float64)[positions]
            cube_rows.append({
                "dataset": "debian_audit_2026",
                "seed": seed,
                "method": method,
                "n_test": len(wanted_event),
                "n_test_events": int(wanted_event.sum()),
                **prospective_metrics(scores, wanted_event),
            })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path,
                        default=[
                            *[RESULTS / f"temporal_native_final_debian_seed{i}.csv"
                              for i in range(5)],
                            *[RESULTS / f"temporal_native_final_swissprot_seed{i}.csv"
                              for i in range(5)],
                        ])
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()

    data = load_results(args.inputs)
    combined_path = RESULTS / "temporal_native_results.csv"
    data.to_csv(combined_path, index=False)
    selected = data[data.selected == 1].copy()
    seeds = sorted(selected.seed.unique().tolist())
    methods = [method for method in METHOD_ORDER if method in set(selected.method)]

    audit_rows: list[dict] = []
    add_debian_audit(audit_rows, methods, seeds)
    audit = pd.DataFrame(audit_rows)
    selected_metrics = selected[["dataset", "seed", "method", "n_test",
                                 "n_test_events", "event_rate"] + METRICS]
    all_metrics = pd.concat([selected_metrics, audit], ignore_index=True, sort=False)
    summary = (all_metrics.groupby(["dataset", "method"])[METRICS]
               .agg(["mean", "std"]).reset_index())
    summary.columns = ["_".join(x).rstrip("_") if isinstance(x, tuple) else x
                       for x in summary.columns]

    rng = np.random.default_rng(args.seed)
    ci_rows, comparison_rows = [], []
    analysis_frames = {
        dataset: selected[selected.dataset == dataset]
        for dataset in sorted(selected.dataset.unique())
    }
    analysis_frames["debian_audit_2026"] = audit
    for dataset, observed_rows in analysis_frames.items():
        boot = paired_hierarchical_bootstrap(dataset, methods, seeds,
                                             args.bootstrap, rng)
        observed = observed_rows.groupby("method")[METRICS].mean()
        for metric in METRICS:
            for method in methods:
                values = boot[metric][method]
                ci_rows.append({
                    "dataset": dataset,
                    "method": method,
                    "metric": metric,
                    "estimate": observed.loc[method, metric],
                    "ci_low": np.quantile(values, 0.025),
                    "ci_high": np.quantile(values, 0.975),
                    "bootstrap_repetitions": args.bootstrap,
                })
        best = observed["ap"].idxmax()
        raw_p = {}
        effects = {}
        for method in methods:
            if method == best:
                continue
            difference = boot["ap"][best] - boot["ap"][method]
            lower = (np.count_nonzero(difference <= 0) + 1) / (len(difference) + 1)
            upper = (np.count_nonzero(difference >= 0) + 1) / (len(difference) + 1)
            pvalue = min(1.0, 2 * min(lower, upper))
            raw_p[method] = pvalue
            effects[method] = (observed.loc[best, "ap"] - observed.loc[method, "ap"],
                               np.quantile(difference, 0.025),
                               np.quantile(difference, 0.975))
        adjusted = holm_adjust(raw_p)
        for method in raw_p:
            effect, low, high = effects[method]
            comparison_rows.append({
                "dataset": dataset,
                "best_method": best,
                "comparator": method,
                "ap_difference": effect,
                "ci_low": low,
                "ci_high": high,
                "p_bootstrap_two_sided": raw_p[method],
                "p_holm": adjusted[method],
            })

    ci = pd.DataFrame(ci_rows)
    comparisons = pd.DataFrame(comparison_rows)
    ci.to_csv(RESULTS / "temporal_native_bootstrap_ci.csv", index=False)
    comparisons.to_csv(RESULTS / "temporal_native_bootstrap_comparisons.csv", index=False)
    summary.to_csv(RESULTS / "temporal_native_summary.csv", index=False)
    audit.to_csv(RESULTS / "temporal_native_debian_audit.csv", index=False)

    lines = []
    lines.append("TEMPORAL NATIVE-PU AUDIT")
    lines.append(f"selected rows: {len(selected)}; seeds: {seeds}; methods: {len(methods)}")
    lines.append("selection: Lee-Liu P/U criterion; test outcomes never used for selection")
    for dataset in sorted(selected.dataset.unique()):
        block = selected[selected.dataset == dataset]
        lines.append(f"\n[{dataset}] events {int(block.n_test_events.iloc[0])}/"
                     f"{int(block.n_test.iloc[0])}")
        aggregate = block.groupby("method")[METRICS].mean().sort_values("ap", ascending=False)
        for method, row in aggregate.iterrows():
            lines.append(f"  {method:18s} AP={row.ap:.5f} AUC={row.auc:.4f} "
                         f"EF1%={row.enrichment_at_1pct:.2f} R1%={row.recall_at_1pct:.3f}")
        alpha = block.groupby("seed")[["alpha_train_bbe", "alpha_val_bbe"]].first()
        lines.append(f"  BBE train mean={alpha.alpha_train_bbe.mean():.4f}; "
                     f"validation mean={alpha.alpha_val_bbe.mean():.4f}")
    (RESULTS / "summary_temporal_native.txt").write_text("\n".join(lines) + "\n",
                                                          encoding="utf8")
    print("\n".join(lines))
    print(f"\nwrote {combined_path} and bootstrap summaries")


if __name__ == "__main__":
    main()
