from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pu_lib import (
    PUData, load_dataset, metrics_from_scores, scores, subsample_to_alpha,
    train_model_v2,
)
from run_prior_estimation_track import (
    bbe_from_scores, common_oof_scores,
)


DATASETS = ("spambase", "adult", "bank", "mushroom")
PRIORS = (0.10, 0.075, 0.05, 0.03, 0.02, 0.01)
METHODS = ("uasn", "wce", "nnpu", "imbpu", "cvir", "bbe_cvir")
LOCKED_CONFIG = {
    "uasn": (1e-3, 0.0),
    "wce": (1e-2, 1e-4),
    "nnpu": (1e-2, 0.0),
    "imbpu": (1e-2, 0.0),
    "cvir": (1e-2, 0.0),
    "bbe_cvir": (1e-3, 0.0),
}
FIELDS = [
    "dataset", "alpha_target", "c", "seed", "method", "base_method", "lr",
    "wd", "epochs", "n_train", "n_labeled_positive", "n_unlabeled",
    "alpha_train_population", "alpha_unlabeled", "alpha_test", "alpha_used",
    "alpha_estimation_ratio", "bacc", "f1", "auc", "auprc", "ap_lift",
    "normalized_ap_lift", "f1_bestthr", "thr_best", "precision", "recall",
    "pred_rate", "ece", "brier", "secs", "status", "error",
]


def make_disjoint_scar(
    x: np.ndarray, y: np.ndarray, c: float, target_alpha: float, seed: int
) -> PUData:
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    n_label = max(1, int(round(c * len(positive))))
    labeled = np.sort(rng.choice(positive, n_label, replace=False))
    positive_u = np.setdiff1d(positive, labeled)
    n_negative_u = int(round(len(positive_u) * (1.0 - target_alpha)
                             / target_alpha))
    if n_negative_u > len(negative):
        raise ValueError(
            f"not enough negatives to construct U at alpha={target_alpha:g}"
        )
    negative_u = rng.choice(negative, n_negative_u, replace=False)
    unlabeled = np.concatenate([positive_u, negative_u])
    rng.shuffle(unlabeled)
    y_u = y[unlabeled]
    s = np.zeros(len(unlabeled), dtype=np.int8)
    e = np.full(len(unlabeled), np.nan)
    return PUData(
        X_p=x[labeled], X_u=x[unlabeled], y_u=y_u,
        alpha_true=float(y_u.mean()), c=n_label / len(positive), s=s,
        e_true=e,
    )


def _completed(path: Path):
    done = set()
    if path.exists():
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("status") == "ok":
                    done.add((row["dataset"], float(row["alpha_target"]),
                              int(row["seed"]), row["method"]))
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--priors", nargs="*", type=float, default=list(PRIORS))
    ap.add_argument("--methods", nargs="*", choices=METHODS, default=list(METHODS))
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--c", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--max-rows", type=int, default=8000)
    ap.add_argument(
        "--out", default=str(ROOT / "results" / "rare_dense_track.csv")
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _completed(out)
    new = not out.exists()
    with out.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        for dataset, alpha_target, seed in itertools.product(
            args.datasets, args.priors, args.seeds
        ):
            print(f"[rare-dense] {dataset} alpha={alpha_target:g} seed={seed}",
                  flush=True)
            try:
                x, y = load_dataset(dataset, max_rows=args.max_rows, seed=seed)
                x, y = subsample_to_alpha(x, y, alpha_target, seed)
                x_train, x_test, y_train, y_test = train_test_split(
                    x, y, test_size=0.25, stratify=y, random_state=seed
                )
                mu, sd = x_train.mean(0), x_train.std(0) + 1e-9
                x_train, x_test = (x_train - mu) / sd, (x_test - mu) / sd
                pu = make_disjoint_scar(
                    x_train, y_train, args.c, alpha_target, seed
                )
                score_p, score_u = common_oof_scores(pu.X_p, pu.X_u, seed)
                alpha_bbe = bbe_from_scores(score_p, score_u)
            except Exception as exc:
                print(f"  CELL FAILED: {type(exc).__name__}: {exc}", flush=True)
                for method in args.methods:
                    key = (dataset, float(alpha_target), int(seed), method)
                    if key not in done:
                        writer.writerow(dict(
                            dataset=dataset, alpha_target=alpha_target, c=args.c,
                            seed=seed, method=method, status="failed",
                            error=f"cell setup: {type(exc).__name__}: {exc}",
                        ))
                fh.flush()
                continue

            for method in args.methods:
                key = (dataset, float(alpha_target), int(seed), method)
                if key in done:
                    continue
                start = time.time()
                base = "cvir" if method == "bbe_cvir" else method
                alpha_used = alpha_bbe if method == "bbe_cvir" else pu.alpha_true
                lr, wd = LOCKED_CONFIG[method]
                row = dict(
                    dataset=dataset, alpha_target=alpha_target, c=args.c,
                    seed=seed, method=method, base_method=base, lr=lr, wd=wd,
                    epochs=args.epochs, n_train=len(y_train),
                    n_labeled_positive=len(pu.X_p), n_unlabeled=len(pu.X_u),
                    alpha_train_population=float(y_train.mean()),
                    alpha_unlabeled=pu.alpha_true,
                    alpha_test=float(y_test.mean()), alpha_used=alpha_used,
                    alpha_estimation_ratio=(alpha_used / pu.alpha_true
                                            if method == "bbe_cvir" else 1.0),
                    status="ok", error="",
                )
                try:
                    model = train_model_v2(
                        pu, base, alpha_used, lr, wd, args.epochs, seed
                    )
                    probability = scores(model, x_test)
                    metrics = metrics_from_scores(
                        probability, y_test, threshold=0.5, extended=True
                    )
                    prevalence = float(y_test.mean())
                    lift = metrics["auprc"] - prevalence
                    row.update(
                        **metrics,
                        ap_lift=lift,
                        normalized_ap_lift=lift / max(1.0 - prevalence, 1e-12),
                        secs=time.time() - start,
                    )
                except Exception as exc:
                    row.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                               secs=time.time() - start)
                    print(f"  {method} FAILED: {row['error']}", flush=True)
                writer.writerow(row); fh.flush()
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
