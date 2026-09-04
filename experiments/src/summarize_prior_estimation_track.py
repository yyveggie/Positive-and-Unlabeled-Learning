from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
KEY = ["separation", "alpha_true", "mechanism", "seed", "estimator"]
EXPECTED_ESTIMATORS = {"BBE", "DEDPUL", "TIcE", "KM1", "KM2"}
EXPECTED_MECHANISMS = {"SCAR", "SAR-attribute", "SAR-gap"}
EXPECTED_SEPARATIONS = {0.5, 1.0, 2.0, 4.0}
EXPECTED_PRIORS = {0.02, 0.05, 0.10, 0.30}
EXPECTED_SEEDS = {0, 1, 2, 3, 4}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pattern",
        default=str(ROOT / "results" / "prior_estimation_seed*.csv"),
    )
    ap.add_argument(
        "--out", default=str(ROOT / "results" / "prior_estimation_track.csv")
    )
    ap.add_argument(
        "--correction-pattern",
        default=str(ROOT / "results" / "prior_tice_corrected_seed*.csv"),
        help="optional audited replacement shards; matching design keys replace "
             "the original rows rather than being averaged or duplicated",
    )
    args = ap.parse_args()

    pattern = Path(args.pattern)
    files = sorted(pattern.parent.glob(pattern.name))
    if not files:
        raise FileNotFoundError(args.pattern)
    frames = [pd.read_csv(path) for path in files]
    data = pd.concat(frames, ignore_index=True)
    correction_pattern = Path(args.correction_pattern)
    correction_files = sorted(
        correction_pattern.parent.glob(correction_pattern.name)
    )
    if correction_files:
        corrected = pd.concat(
            [pd.read_csv(path) for path in correction_files], ignore_index=True
        )
        corrected_keys = pd.MultiIndex.from_frame(corrected[KEY])
        original_keys = pd.MultiIndex.from_frame(data[KEY])
        data = pd.concat(
            [data.loc[~original_keys.isin(corrected_keys)], corrected],
            ignore_index=True,
        )

    duplicates = int(data.duplicated(KEY).sum())
    failures = data.loc[data["status"] != "ok", KEY + ["status", "error"]]
    if duplicates:
        raise RuntimeError(f"found {duplicates} duplicate design keys")
    if len(failures):
        raise RuntimeError(f"found {len(failures)} failed rows:\n{failures}")
    if set(data["estimator"]) != EXPECTED_ESTIMATORS:
        raise RuntimeError(f"unexpected estimator set: {set(data['estimator'])}")
    if set(data["mechanism"]) != EXPECTED_MECHANISMS:
        raise RuntimeError(f"unexpected mechanism set: {set(data['mechanism'])}")
    if set(data["separation"]) != EXPECTED_SEPARATIONS:
        raise RuntimeError(f"unexpected separation set: {set(data['separation'])}")
    if set(data["alpha_true"]) != EXPECTED_PRIORS:
        raise RuntimeError(f"unexpected prior set: {set(data['alpha_true'])}")
    if set(data["seed"]) != EXPECTED_SEEDS:
        raise RuntimeError(f"unexpected seed set: {set(data['seed'])}")

    levels = {
        "separation": sorted(data["separation"].unique()),
        "alpha_true": sorted(data["alpha_true"].unique()),
        "mechanism": sorted(data["mechanism"].unique()),
        "seed": sorted(data["seed"].unique()),
        "estimator": sorted(data["estimator"].unique()),
    }
    expected = (len(EXPECTED_SEPARATIONS) * len(EXPECTED_PRIORS)
                * len(EXPECTED_MECHANISMS) * len(EXPECTED_SEEDS)
                * len(EXPECTED_ESTIMATORS))
    if len(data) != expected:
        raise RuntimeError(
            f"incomplete factorial: observed {len(data)} rows, expected {expected}; "
            f"levels={levels}"
        )

    data = data.sort_values(KEY).reset_index(drop=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out, index=False)

    by_cell = (
        data.groupby(["separation", "alpha_true", "mechanism", "estimator"],
                     as_index=False)
        .agg(
            mean_abs_error=("abs_error", "mean"),
            sd_abs_error=("abs_error", "std"),
            mean_signed_error=("signed_error", "mean"),
            median_relative_error=("relative_error", "median"),
            mean_alpha_hat=("alpha_hat", "mean"),
            sd_alpha_hat=("alpha_hat", "std"),
            mean_oracle_auc=("oracle_auc", "mean"),
            n=("seed", "size"),
        )
    )
    by_cell.to_csv(out.with_name("prior_estimation_by_cell.csv"), index=False)

    by_estimator = (
        data.groupby(["mechanism", "estimator"], as_index=False)
        .agg(
            mean_abs_error=("abs_error", "mean"),
            median_abs_error=("abs_error", "median"),
            mean_signed_error=("signed_error", "mean"),
            median_relative_error=("relative_error", "median"),
            n=("seed", "size"),
        )
    )
    by_estimator.to_csv(out.with_name("prior_estimation_by_estimator.csv"), index=False)
    print(f"wrote {out} ({len(data)} complete rows)")


if __name__ == "__main__":
    main()
