from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEDPUL = ROOT / "external" / "dedpul"
sys.path.insert(0, str(DEDPUL))

from algorithms import estimate_diff, estimate_poster_em
from KMPE import wrapper as kmpe_wrapper
from TIcE import min_max_scale, tice, tice_c_to_alpha


ESTIMATORS = ("BBE", "DEDPUL", "TIcE", "KM1", "KM2")
FIELDS = (
    "separation", "alpha_true", "mechanism", "seed", "estimator",
    "alpha_hat", "abs_error", "signed_error", "relative_error",
    "oracle_auc", "n_positive", "n_unlabeled", "seconds", "status", "error",
)


def _positive_posterior(x0: np.ndarray, alpha: float, separation: float) -> np.ndarray:
    prior_log_odds = logit(np.clip(alpha, 1e-8, 1 - 1e-8))
    return expit(prior_log_odds + separation * x0)


def make_case_control_pu(
    separation: float,
    alpha: float,
    mechanism: str,
    seed: int,
    n_positive: int,
    n_unlabeled: int,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if mechanism not in {"SCAR", "SAR-attribute", "SAR-gap"}:
        raise ValueError(f"unknown mechanism: {mechanism}")
    rng = np.random.default_rng(seed)
    mu_pos = np.zeros(dimension)
    mu_neg = np.zeros(dimension)
    mu_pos[0] = separation / 2
    mu_neg[0] = -separation / 2

    y_u = (rng.random(n_unlabeled) < alpha).astype(np.int8)
    x_u = rng.normal(size=(n_unlabeled, dimension))
    x_u += np.where(y_u[:, None] == 1, mu_pos, mu_neg)

    pool_size = max(20 * n_positive, 5000)
    pool = rng.normal(size=(pool_size, dimension)) + mu_pos
    if mechanism == "SCAR":
        weights = np.ones(pool_size)
    elif mechanism == "SAR-attribute":


        z = (pool[:, 0] - pool[:, 0].mean()) / (pool[:, 0].std() + 1e-12)
        weights = 0.02 + 0.98 * expit(3.0 * z)
    else:


        weights = 0.02 + 0.98 * _positive_posterior(pool[:, 0], alpha, separation) ** 2
    weights = weights / weights.sum()
    chosen = rng.choice(pool_size, size=n_positive, replace=False, p=weights)
    x_p = pool[chosen]

    scaler = StandardScaler().fit(np.vstack([x_p, x_u]))
    x_p = scaler.transform(x_p)
    x_u = scaler.transform(x_u)
    oracle_auc = float(roc_auc_score(y_u, x_u[:, 0]))
    return x_p.astype(np.float64), x_u.astype(np.float64), y_u, oracle_auc


def common_oof_scores(x_p: np.ndarray, x_u: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.vstack([x_p, x_u])


    pu_target = np.concatenate([np.zeros(len(x_p), dtype=int), np.ones(len(x_u), dtype=int)])
    oof_u = np.empty(len(x), dtype=np.float64)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, va in cv.split(x, pu_target):
        model = LogisticRegression(
            solver="lbfgs", max_iter=2000, class_weight="balanced",
            random_state=seed,
        )
        model.fit(x[tr], pu_target[tr])
        oof_u[va] = model.predict_proba(x[va])[:, 1]
    score_p = 1.0 - oof_u[: len(x_p)]
    score_u = 1.0 - oof_u[len(x_p) :]
    return np.clip(score_p, 1e-7, 1 - 1e-7), np.clip(score_u, 1e-7, 1 - 1e-7)


def bbe_from_scores(score_p: np.ndarray, score_u: np.ndarray, delta: float = 0.1) -> float:
    p = np.sort(np.asarray(score_p))[::-1]
    u = np.sort(np.asarray(score_u))[::-1]
    upper, estimates = [], []
    j = 0
    for i in range(2, len(u)):
        threshold = u[i]
        if i < len(u) - 1 and threshold <= u[i + 1]:
            continue
        while j < len(p) and p[j] >= threshold:
            j += 1
        if j <= 1:
            continue
        estimate = i * len(p) / (j * len(u))
        temp = math.sqrt(math.log(4 / delta) / (2 * len(u))) + math.sqrt(
            math.log(4 / delta) / (2 * len(p))
        )
        bound = temp * 1.01 / (j / len(p))
        estimates.append(estimate)
        upper.append(estimate + bound)
    if not upper:
        return 0.0
    return float(np.clip(estimates[int(np.argmin(upper))], 0.0, 1.0))


def dedpul_from_scores(score_p: np.ndarray, score_u: np.ndarray) -> float:

    pred_u = np.concatenate([1.0 - score_p, 1.0 - score_u])
    target = np.concatenate([np.zeros(len(score_p), dtype=int), np.ones(len(score_u), dtype=int)])
    diff = estimate_diff(
        pred_u, target, bw_mix=0.05, bw_pos=0.1, kde_mode="logit",
        threshold="mid", tune=False, MT=True, MT_coef=0.25,
        decay_MT_coef=False, k_neighbours=None,
    )
    alpha_negative, _ = estimate_poster_em(
        diff=diff, mode="dedpul", alpha=None, converge=True,
        nonconverge=True, disp=False,
    )
    return float(np.clip(1.0 - alpha_negative, 0.0, 1.0))


def tice_from_features(x_p: np.ndarray, x_u: np.ndarray, seed: int) -> float:
    x = min_max_scale(np.vstack([x_p, x_u]))
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    labels = np.concatenate([np.ones(len(x_p), dtype=int), np.zeros(len(x_u), dtype=int)])
    rng = np.random.default_rng(seed)
    folds = rng.integers(0, 10, size=len(x))
    c_hat = tice(
        x, labels, 10, folds, delta=0.2, nbIterations=2,
        maxSplits=500, useMostPromisingOnly=False, minT=10, n_splits=40,
    )[0]
    gamma_u = len(x_u) / len(x)
    if c_hat <= 0:
        return 1.0


    alpha_negative = tice_c_to_alpha(c_hat, gamma_u)
    return float(np.clip(1.0 - alpha_negative, 0.0, 1.0))


def km_from_features(x_p: np.ndarray, x_u: np.ndarray, which: str) -> float:
    kwargs = dict(
        X_mixture=x_u, X_component=x_p, disp=False,
        lambda_lower_bound=1.0, lambda_upper_bound=8.0,
    )
    if which == "KM1":
        ans = kmpe_wrapper(KM_1=True, KM_2=False, **kwargs)
    elif which == "KM2":
        ans = kmpe_wrapper(KM_1=False, KM_2=True, **kwargs)
    else:
        raise ValueError(which)
    return float(np.clip(ans, 0.0, 1.0))


def _read_completed(path: Path) -> set[tuple[float, float, str, int, str]]:
    done: set[tuple[float, float, str, int, str]] = set()
    if not path.exists():
        return done
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") == "ok":
                done.add((float(row["separation"]), float(row["alpha_true"]),
                          row["mechanism"], int(row["seed"]), row["estimator"]))
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "prior_estimation_track.csv"))
    ap.add_argument("--separations", nargs="*", type=float, default=[0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--alphas", nargs="*", type=float, default=[0.02, 0.05, 0.10, 0.30])
    ap.add_argument("--mechanisms", nargs="*", default=["SCAR", "SAR-attribute", "SAR-gap"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--estimators", nargs="*", default=list(ESTIMATORS), choices=ESTIMATORS)
    ap.add_argument("--n-positive", type=int, default=200)
    ap.add_argument("--n-unlabeled", type=int, default=600)
    ap.add_argument("--dimension", type=int, default=10)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    completed = _read_completed(out)
    new = not out.exists()
    with out.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        for sep in args.separations:
            for alpha in args.alphas:
                for mechanism in args.mechanisms:
                    for seed in args.seeds:
                        cell_seed = int(seed + round(sep * 1000) * 100 + round(alpha * 1000) * 10_000)
                        x_p, x_u, y_u, oracle_auc = make_case_control_pu(
                            sep, alpha, mechanism, cell_seed, args.n_positive,
                            args.n_unlabeled, args.dimension,
                        )
                        score_p = score_u = None
                        if any(e in {"BBE", "DEDPUL"} for e in args.estimators):
                            score_p, score_u = common_oof_scores(x_p, x_u, cell_seed)
                        print(
                            f"[prior] sep={sep:g} alpha={alpha:g} {mechanism} seed={seed}",
                            flush=True,
                        )
                        for estimator in args.estimators:
                            key = (float(sep), float(alpha), mechanism, int(seed), estimator)
                            if key in completed:
                                continue
                            start = time.time()
                            status, error = "ok", ""
                            try:
                                if estimator == "BBE":
                                    alpha_hat = bbe_from_scores(score_p, score_u)
                                elif estimator == "DEDPUL":
                                    alpha_hat = dedpul_from_scores(score_p, score_u)
                                elif estimator == "TIcE":
                                    alpha_hat = tice_from_features(x_p, x_u, cell_seed)
                                else:
                                    alpha_hat = km_from_features(x_p, x_u, estimator)
                                if not np.isfinite(alpha_hat):
                                    raise FloatingPointError("non-finite alpha estimate")
                            except Exception as exc:
                                alpha_hat = float("nan")
                                status = "failed"
                                error = f"{type(exc).__name__}: {exc}"
                                traceback.print_exc()
                            row = dict(
                                separation=sep, alpha_true=alpha, mechanism=mechanism,
                                seed=seed, estimator=estimator,
                                alpha_hat=alpha_hat,
                                abs_error=abs(alpha_hat - alpha) if np.isfinite(alpha_hat) else "",
                                signed_error=alpha_hat - alpha if np.isfinite(alpha_hat) else "",
                                relative_error=(alpha_hat - alpha) / alpha if np.isfinite(alpha_hat) else "",
                                oracle_auc=oracle_auc, n_positive=len(x_p), n_unlabeled=len(x_u),
                                seconds=time.time() - start, status=status, error=error,
                            )
                            writer.writerow(row)
                            fh.flush()
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
