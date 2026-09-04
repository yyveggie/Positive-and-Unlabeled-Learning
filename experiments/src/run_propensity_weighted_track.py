from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "methods"))

from pu_lib import load_dataset, make_pu, metrics_from_scores


DATASETS = ["breast", "spambase", "mushroom", "adult", "bank"]
MECHANISMS = ["scar", "sar_indep", "sar_attr", "sar_gap", "sar_extreme"]
CS = [0.3, 0.5]
FIELDS = [
    "dataset", "mechanism", "c", "seed", "method", "posterior_model",
    "alpha_true", "c_realized", "bacc", "f1", "auc", "auprc",
    "f1_bestthr", "thr_best", "precision", "recall", "pred_rate", "ece",
    "brier", "k_hat", "e_min_labeled", "e_median_labeled",
    "e_max_labeled", "propensity_mae_positive", "propensity_spearman_positive",
    "risk_steps", "secs", "status", "error",
]


def oof_label_posterior(x: np.ndarray, s: np.ndarray, seed: int) -> np.ndarray:
    if np.unique(s).size != 2:
        raise ValueError("both labeled and unlabeled observations are required")
    n_splits = min(5, int(s.sum()), int((1 - s).sum()))
    if n_splits < 2:
        raise ValueError("insufficient observations for cross-fitting")
    out = np.empty(len(x), dtype=np.float64)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(cv.split(x, s)):
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=150,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=seed * 17 + fold,
        )
        model.fit(x[tr], s[tr])
        out[va] = model.predict_proba(x[va])[:, 1]
    return np.clip(out, 1e-8, 1 - 1e-8)


def fit_propensity_weighted_logistic(
    x: np.ndarray,
    s: np.ndarray,
    e_hat: np.ndarray,
    seed: int,
    steps: int = 1200,
    lr: float = 0.02,
    l2: float = 1e-3,
) -> tuple[torch.nn.Module, int]:
    torch.manual_seed(seed)
    xt = torch.as_tensor(x, dtype=torch.float32)
    st = torch.as_tensor(s, dtype=torch.float32)
    et = torch.as_tensor(e_hat, dtype=torch.float32)
    model = torch.nn.Linear(x.shape[1], 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = float("inf")
    stale = 0
    used = steps
    for step in range(1, steps + 1):
        z = model(xt).squeeze(1)
        positive_loss = torch.nn.functional.softplus(-z)
        negative_loss = torch.nn.functional.softplus(z)
        inverse = st / et
        risk = (inverse * positive_loss + (1.0 - inverse) * negative_loss).mean()
        penalty = 0.5 * l2 * model.weight.square().sum()
        loss = risk + penalty
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite propensity-weighted risk")
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        value = float(loss.detach())
        if value < best - 1e-7:
            best, stale = value, 0
        else:
            stale += 1
        if step >= 300 and stale >= 150:
            used = step
            break
    return model.eval(), used


def _completed(path: Path) -> set[tuple[str, str, float, int]]:
    done: set[tuple[str, str, float, int]] = set()
    if not path.exists():
        return done
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") == "ok":
                done.add((row["dataset"], row["mechanism"], float(row["c"]),
                          int(row["seed"])))
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--mechanisms", nargs="*", default=MECHANISMS)
    ap.add_argument("--cs", nargs="*", type=float, default=CS)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument(
        "--out", default=str(ROOT / "results" / "propensity_weighted_track.csv")
    )
    ap.add_argument("--risk-steps", type=int, default=1200)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _completed(out)
    new = not out.exists()
    with out.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        for dataset, mechanism, c, seed in itertools.product(
            args.datasets, args.mechanisms, args.cs, args.seeds
        ):
            key = (dataset, mechanism, float(c), int(seed))
            if key in done:
                continue
            print(f"[pg-ipw] {dataset} {mechanism} c={c:g} seed={seed}", flush=True)
            start = time.time()
            row = dict(
                dataset=dataset, mechanism=mechanism, c=c, seed=seed,
                method="pg_ipw", posterior_model="HGB-5fold-OOF", status="ok",
                error="",
            )
            try:
                x, y = load_dataset(dataset, seed=seed)
                x_train, x_tmp, y_train, y_tmp = train_test_split(
                    x, y, test_size=0.4, stratify=y, random_state=seed
                )
                _, x_test, _, y_test = train_test_split(
                    x_tmp, y_tmp, test_size=0.5, stratify=y_tmp,
                    random_state=seed,
                )
                mu = x_train.mean(axis=0)
                sd = x_train.std(axis=0) + 1e-9
                x_train = (x_train - mu) / sd
                x_test = (x_test - mu) / sd
                pu = make_pu(x_train, y_train, c, mechanism, seed)
                s = pu.s.astype(np.int8)

                h_oof = oof_label_posterior(x_train, s, seed)
                k_hat = float(h_oof.max())
                e_hat = np.sqrt(k_hat * h_oof)


                e_for_risk = np.clip(e_hat, 1e-3, 1.0)
                model, used = fit_propensity_weighted_logistic(
                    x_train, s, e_for_risk, seed, steps=args.risk_steps
                )
                with torch.no_grad():
                    score = torch.sigmoid(
                        model(torch.as_tensor(x_test, dtype=torch.float32)).squeeze(1)
                    ).numpy()

                labeled_e = e_hat[s == 1]
                positive = y_train == 1
                true_e = pu.e_true[positive]
                est_e = e_hat[positive]
                if np.std(true_e) < 1e-12 or np.std(est_e) < 1e-12:
                    rho = np.nan
                else:
                    rho = spearmanr(true_e, est_e).statistic
                row.update(
                    alpha_true=pu.alpha_true,
                    c_realized=pu.c,
                    **metrics_from_scores(score, y_test, 0.5, extended=True),
                    k_hat=k_hat,
                    e_min_labeled=float(labeled_e.min()),
                    e_median_labeled=float(np.median(labeled_e)),
                    e_max_labeled=float(labeled_e.max()),
                    propensity_mae_positive=float(np.mean(np.abs(est_e - true_e))),
                    propensity_spearman_positive=float(rho),
                    risk_steps=used,
                    secs=time.time() - start,
                )
            except Exception as exc:
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                           secs=time.time() - start)
                print(f"  FAILED: {row['error']}", flush=True)
            writer.writerow(row)
            fh.flush()
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
