from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "methods"))
from official_adapters import OFFICIAL_GRID, train_official
from pu_lib import PUData, bbe_alpha, criteria_from_scores, scores, train_model_v2


HERE = Path(__file__).resolve().parents[1]
DATA_DIR = HERE / "features" / "native_temporal"
DEFAULT_OUT = HERE / "results" / "temporal_native_results.csv"
DEFAULT_PRED_DIR = HERE / "results" / "temporal_native_predictions"

SELF_METHODS = ["uasn", "wce", "upu_bbe", "nnpu_bbe", "imbpu_bbe", "cvir_bbe"]
OFFICIAL_METHODS = ["vpu_off", "distpu_bbe_off", "holistic_off"]
HP = [(lr, wd) for lr in (1e-2, 1e-3) for wd in (0.0, 1e-4)]
EPOCHS = 50
FIELDS = [
    "dataset", "seed", "method", "implementation", "prior_source",
    "selection_criterion", "selected",
    "lr", "wd", "cfg", "alpha_train_bbe", "alpha_val_bbe",
    "n_train_p", "n_train_u", "n_val_p", "n_val_u", "n_test", "n_test_events",
    "event_rate", "crit_nnpurisk", "crit_leeliu", "crit_puauc",
    "auc", "ap", "recall_at_1pct", "precision_at_1pct", "enrichment_at_1pct",
    "recall_at_5pct", "precision_at_5pct", "enrichment_at_5pct", "secs",
]


@dataclass
class TemporalCohort:
    name: str
    X: np.ndarray
    entity_ids: np.ndarray
    p0: np.ndarray
    val_event: np.ndarray
    val_control: np.ndarray
    test_mask: np.ndarray
    test_event: np.ndarray


def load_cohort(name: str) -> TemporalCohort:
    if name == "debian":
        d = np.load(DATA_DIR / "debian_vulnerability_temporal.npz", allow_pickle=True)
        X, labels = d["X"].astype(np.float64), d["labels"].astype(bool)
        entity_ids = d["packages"].astype(str)
        present = d["present"].astype(bool)
        known_by = d["known_by"].astype(bool)
        p0, val, test = labels[:, 0], labels[:, 1], labels[:, 2]


        val_control = present[:, 1] & (~known_by[:, 1])
        test_mask = present[:, 2] & (~known_by[:, 1])
    elif name == "swissprot":
        d = np.load(DATA_DIR / "swissprot_nucleus_temporal.npz", allow_pickle=True)
        X, labels = d["X"].astype(np.float64), d["labels"].astype(bool)
        entity_ids = d["accessions"].astype(str)
        present = d["present"].astype(bool)
        p0, val, test = labels[:, 0], labels[:, 1], labels[:, 2]
        val_control = (~p0) & (~val) & present[:, 1]
        test_mask = (~p0) & (~val) & present[:, 2]
    else:
        raise ValueError(name)


    mean, std = X.mean(axis=0), X.std(axis=0)
    keep = std > 1e-10
    X = (X[:, keep] - mean[keep]) / std[keep]
    val_event = val
    test_event = test & test_mask
    if val_event.sum() < 5 or test_event.sum() < 5:
        raise RuntimeError(
            f"{name} has too few temporal events: val={val_event.sum()}, test={test_event.sum()}"
        )
    return TemporalCohort(
        name, X, entity_ids, p0, val_event, val_control, test_mask, test_event
    )


def make_pu(Xp: np.ndarray, Xu: np.ndarray) -> PUData:


    return PUData(
        X_p=Xp,
        X_u=Xu,
        y_u=np.zeros(len(Xu), dtype=np.int8),
        alpha_true=float("nan"),
        c=float("nan"),
    )


def prospective_metrics(probability: np.ndarray, event: np.ndarray) -> dict[str, float]:
    probability = np.asarray(probability, dtype=np.float64)
    event = np.asarray(event, dtype=bool)
    n, positives = len(event), int(event.sum())
    if positives == 0 or positives == n:
        raise ValueError("prospective metrics require both events and censored controls")
    rate = positives / n
    order = np.argsort(-probability, kind="mergesort")
    out = {
        "event_rate": rate,
        "auc": float(roc_auc_score(event, probability)),
        "ap": float(average_precision_score(event, probability)),
    }
    for fraction, label in ((0.01, "1pct"), (0.05, "5pct")):
        k = max(1, int(math.ceil(fraction * n)))
        hits = int(event[order[:k]].sum())
        precision = hits / k
        out[f"recall_at_{label}"] = hits / positives
        out[f"precision_at_{label}"] = precision
        out[f"enrichment_at_{label}"] = precision / rate
    return out


def run_dataset(writer, cohort: TemporalCohort, seed: int, smoke: bool,
                self_methods: list[str], official_methods: list[str],
                prediction_dir: Path) -> None:
    X = cohort.X
    pu_train = make_pu(X[cohort.p0], X[~cohort.p0])
    pu_val = make_pu(X[cohort.val_event], X[cohort.val_control])
    alpha_train = bbe_alpha(pu_train, seed)
    alpha_val = bbe_alpha(pu_val, seed + 10_000)
    y_test = cohort.test_event[cohort.test_mask].astype(np.int8)
    X_test = X[cohort.test_mask]
    print(
        f"[{cohort.name} seed={seed}] P/U={len(pu_train.X_p)}/{len(pu_train.X_u)}, "
        f"val P/U={len(pu_val.X_p)}/{len(pu_val.X_u)}, test events="
        f"{y_test.sum()}/{len(y_test)}, BBE train={alpha_train:.4f}, "
        f"val={alpha_val:.4f}",
        flush=True,
    )

    common = {
        "dataset": cohort.name,
        "seed": seed,
        "alpha_train_bbe": round(alpha_train, 6),
        "alpha_val_bbe": round(alpha_val, 6),
        "n_train_p": len(pu_train.X_p),
        "n_train_u": len(pu_train.X_u),
        "n_val_p": len(pu_val.X_p),
        "n_val_u": len(pu_val.X_u),
        "n_test": len(y_test),
        "n_test_events": int(y_test.sum()),
    }

    def assess(method: str, proba, seconds: float, implementation: str,
               prior_source: str, lr="", wd="", cfg="") -> dict:
        sp = proba(pu_val.X_p)
        su = proba(pu_val.X_u)
        crit = criteria_from_scores(sp, su, alpha=alpha_val)
        return {
            **common,
            "method": method,
            "implementation": implementation,
            "prior_source": prior_source,
            "selection_criterion": "Lee-Liu (P/U only)",
            "selected": 0,
            "lr": lr,
            "wd": wd,
            "cfg": cfg,
            "crit_nnpurisk": round(crit["crit_nnpurisk"], 6),
            "crit_leeliu": round(crit["crit_leeliu"], 6),
            "crit_puauc": round(crit["crit_puauc"], 6),
            "event_rate": "",
            "auc": "",
            "ap": "",
            "recall_at_1pct": "",
            "precision_at_1pct": "",
            "enrichment_at_1pct": "",
            "recall_at_5pct": "",
            "precision_at_5pct": "",
            "enrichment_at_5pct": "",
            "secs": round(seconds, 2),
            "_proba": proba,
        }

    def emit_selected(candidates: list[dict]) -> None:
        selected = max(range(len(candidates)), key=lambda i: candidates[i]["crit_leeliu"])
        candidates[selected]["selected"] = 1
        selected_row = candidates[selected]


        cohort_scores = np.asarray(selected_row["_proba"](X), dtype=np.float32)
        test_scores = cohort_scores[cohort.test_mask]
        selected_row.update({
            key: round(value, 6)
            for key, value in prospective_metrics(test_scores, y_test).items()
        })
        if not smoke:
            prediction_dir.mkdir(parents=True, exist_ok=True)
            row = selected_row
            path = prediction_dir / f"{cohort.name}_seed{seed}_{row['method']}.npz"
            np.savez_compressed(
                path,
                entity_ids=cohort.entity_ids[cohort.test_mask],
                event=y_test,
                score=test_scores,
                cohort_entity_ids=cohort.entity_ids,
                cohort_score=cohort_scores,
                method=row["method"],
                dataset=cohort.name,
                seed=seed,
                config=row["cfg"] or f"lr={row['lr']}|wd={row['wd']}",
                selection_criterion=row["selection_criterion"],
            )
        for row in candidates:
            row.pop("_proba")
        writer.writerows(candidates)

    methods = self_methods
    for method in methods:
        base = method[:-4] if method.endswith("_bbe") else method
        prior_source = "none" if base in ("uasn", "wce") else "BBE(t0 P/U)"
        candidates = []
        for lr, wd in (HP[:1] if smoke else HP):
            started = time.time()
            model = train_model_v2(
                pu_train, base, alpha_train, lr, wd,
                epochs=5 if smoke else EPOCHS, seed=seed,
            )
            candidates.append(assess(
                method,
                lambda data, fitted=model: scores(fitted, data),
                time.time() - started,
                "loss-level reproduction",
                prior_source,
                lr=lr,
                wd=wd,
            ))
        emit_selected(candidates)

    official = official_methods
    for method in official:
        adapter = "distpu_off" if method == "distpu_bbe_off" else method
        prior_source = "BBE(t0 P/U)" if method == "distpu_bbe_off" else "none"
        candidates = []
        grid = OFFICIAL_GRID[adapter][:1] if smoke else OFFICIAL_GRID[adapter]
        for cfg in grid:
            started = time.time()
            kw = ({"epochs": 5, "val_iterations": 3}
                  if smoke and adapter == "vpu_off" else {})
            if smoke and adapter == "distpu_off":
                kw = {"warm_epochs": 3, "pu_epochs": 3}
            if smoke and adapter == "holistic_off":
                kw = {"ft_epochs": 3}
            predictor = train_official(
                adapter, pu_train, pu_val, alpha_train, seed, cfg, **kw
            )
            candidates.append(assess(
                method,
                predictor.proba,
                time.time() - started,
                "authors' code adapter",
                prior_source,
                cfg="|".join(f"{key}={value}" for key, value in cfg.items()),
            ))
        emit_selected(candidates)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["debian", "swissprot"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--methods", nargs="*", default=SELF_METHODS)
    parser.add_argument("--official", nargs="*", default=OFFICIAL_METHODS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--prediction-dir", type=Path, default=DEFAULT_PRED_DIR)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    new = not args.out.exists()
    with args.out.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        for dataset in args.datasets:
            cohort = load_cohort(dataset)
            for seed in ([0] if args.smoke else args.seeds):
                run_dataset(
                    writer, cohort, seed, args.smoke, args.methods, args.official,
                    args.prediction_dir,
                )
                fh.flush()
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
