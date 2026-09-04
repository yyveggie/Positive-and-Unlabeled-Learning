from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from splice_generated import splice

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
KEY = ["dataset", "seed", "method"]
METHODS = {"uasn_e2e", "nnpu_e2e", "holistic_e2e"}
DATASETS = {"cifar10", "fashionmnist"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs", nargs="*",
        default=[
            str(ROOT / "results" / "end_to_end_cifar.csv"),
            str(ROOT / "results" / "end_to_end_fashion.csv"),
        ],
    )
    ap.add_argument(
        "--out", default=str(ROOT / "results" / "end_to_end_vision_track.csv")
    )
    args = ap.parse_args()
    files = [Path(p) for p in args.inputs]
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError(missing)
    data = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    if data.duplicated(KEY).any():
        raise RuntimeError("duplicate vision design keys")
    failures = data[data.status != "ok"]
    if len(failures):
        raise RuntimeError(f"{len(failures)} failed end-to-end runs")
    if len(data) != len(DATASETS) * len(METHODS) * 3:
        raise RuntimeError(f"incomplete end-to-end grid: {len(data)} rows")
    if set(data.dataset) != DATASETS or set(data.method) != METHODS:
        raise RuntimeError("unexpected end-to-end factor levels")
    if set(data.seed) != {0, 1, 2}:
        raise RuntimeError(f"unexpected seed set: {set(data.seed)}")
    cell_sizes = data.groupby(["dataset", "method"]).size()
    if not (cell_sizes == 3).all():
        raise RuntimeError(f"unbalanced dataset-method cells:\n{cell_sizes}")
    expected_n = {"cifar10": 50000, "fashionmnist": 60000}
    expected_p = {"cifar10": 4000, "fashionmnist": 4800}
    expected_u = {"cifar10": 46000, "fashionmnist": 55200}
    expected_backbone = {
        "cifar10": "HolisticPU-CNN7",
        "fashionmnist": "HolisticPU-LeNet",
    }
    expected_lr = {"cifar10": 0.0015, "fashionmnist": 0.002}
    for dataset, n in expected_n.items():
        part = data.loc[data.dataset == dataset]
        observed = set(part.n_train)
        if observed != {n}:
            raise RuntimeError(f"{dataset} is not the complete training set: {observed}")
        if set(part.n_labeled_positive) != {expected_p[dataset]}:
            raise RuntimeError(f"unexpected {dataset} labeled-positive count")
        if set(part.n_unlabeled) != {expected_u[dataset]}:
            raise RuntimeError(f"unexpected {dataset} unlabeled count")
        if set(part.backbone) != {expected_backbone[dataset]}:
            raise RuntimeError(f"unexpected {dataset} backbone: {set(part.backbone)}")
        if not (part.learning_rate - expected_lr[dataset]).abs().lt(1e-12).all():
            raise RuntimeError(f"unexpected {dataset} learning rate")
    if not (data.c - 0.2).abs().lt(1e-12).all():
        raise RuntimeError("vision label frequency is not c=0.2")
    if set(data.batch_size) != {64}:
        raise RuntimeError(f"unexpected batch-size budget: {set(data.batch_size)}")
    if not (data.alpha_unlabeled - (8 / 23)).abs().lt(1e-12).all():
        raise RuntimeError("unexpected unlabeled-pool prior")
    baseline = data.method != "holistic_e2e"
    if set(data.loc[baseline, "epochs"]) != {20}:
        raise RuntimeError("baseline epoch budget changed")
    holistic = data.method == "holistic_e2e"
    if (set(data.loc[holistic, "epochs"]) != {20}
            or set(data.loc[holistic, "warm_epochs"]) != {8}
            or set(data.loc[holistic, "finetune_epochs"]) != {12}):
        raise RuntimeError("HolisticPU phase budgets changed")
    data = data.sort_values(KEY).reset_index(drop=True)


    baseline = data.method != "holistic_e2e"
    holistic = data.method == "holistic_e2e"
    out = Path(args.out)
    data.to_csv(out, index=False)
    metrics = [
        "bacc", "f1", "auc", "auprc", "f1_bestthr", "thr_best",
        "pred_rate", "ece", "brier",
    ]
    if not np.isfinite(data[metrics + ["secs"]].to_numpy(float)).all():
        raise RuntimeError("non-finite end-to-end metric or runtime")
    if not data[metrics].apply(lambda column: column.between(0, 1).all()).all():
        raise RuntimeError("end-to-end metric outside [0,1]")
    if not (data.secs > 0).all():
        raise RuntimeError("non-positive end-to-end runtime")
    if data.loc[holistic, ["selected_positive_rate", "selected_positive_f1"]].isna().any().any():
        raise RuntimeError("missing HolisticPU selection diagnostics")
    if data.loc[baseline, ["selected_positive_rate", "selected_positive_f1"]].notna().any().any():
        raise RuntimeError("selection diagnostics unexpectedly populated for baselines")
    summary = data.groupby(["dataset", "method"], as_index=False).agg(
        **{f"mean_{m}": (m, "mean") for m in metrics},
        **{f"sd_{m}": (m, "std") for m in metrics},
        mean_selected_positive_rate=("selected_positive_rate", "mean"),
        mean_selected_positive_f1=("selected_positive_f1", "mean"),
        n=("seed", "size"),
    )
    summary.to_csv(out.with_name("end_to_end_vision_summary.csv"), index=False)


    dataset_label = {"cifar10": "CIFAR-10", "fashionmnist": "Fashion-MNIST"}
    method_label = {
        "uasn_e2e": "$U$-as-negative",
        "nnpu_e2e": "nnPU",
        "holistic_e2e": "HolisticPU-style",
    }
    tex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Raw-pixel end-to-end results (descriptive means over three "
        r"seeds). AP is average precision, $F_1^{\ast}$ uses the diagnostic "
        r"best test threshold; selection rate is the fraction of $U$ assigned "
        r"positive, and selection $F_1$ evaluates that HolisticPU-style set "
        r"only after training. Individual seed values are "
        r"retained in the row-level results file.}",
        r"\label{tab:supp-end-to-end}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Dataset & Objective & bacc & AP & $F_1^{\ast}$ & ECE & sel. rate & sel. $F_1$ \\",
        r"\midrule",
    ]
    for dataset in ["cifar10", "fashionmnist"]:
        for method in ["uasn_e2e", "nnpu_e2e", "holistic_e2e"]:
            row = summary[(summary.dataset == dataset)
                          & (summary.method == method)].iloc[0]
            selection_rate = (f"{row.mean_selected_positive_rate:.3f}"
                              if method == "holistic_e2e" else "---")
            selection_f1 = (f"{row.mean_selected_positive_f1:.3f}"
                            if method == "holistic_e2e" else "---")
            tex.append(
                f"{dataset_label[dataset]} & {method_label[method]} & "
                f"{row.mean_bacc:.3f} & {row.mean_auprc:.3f} & "
                f"{row.mean_f1_bestthr:.3f} & {row.mean_ece:.3f} & "
                f"{selection_rate} & {selection_f1} \\\\"
            )
        if dataset == "cifar10":
            tex.append(r"\addlinespace")
    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    splice("e2e", "\n".join(tex),
           "experiments/src/summarize_end_to_end_vision_track.py")

    wide = data.pivot(index=["dataset", "seed"], columns="method",
                      values=metrics)
    rows = []
    for dataset in sorted(DATASETS):
        part = wide.loc[dataset]
        for method in sorted(METHODS - {"uasn_e2e"}):
            for metric in metrics:
                difference = part[(metric, method)] - part[(metric, "uasn_e2e")]
                rows.append(dict(
                    dataset=dataset, method=method, reference="uasn_e2e",
                    metric=metric, mean_difference=difference.mean(),
                    sd_difference=difference.std(ddof=1), n_pairs=len(difference),
                ))
    pd.DataFrame(rows).to_csv(
        out.with_name("end_to_end_vision_paired.csv"), index=False
    )
    print(
        f"wrote {out} ({len(data)} complete full-dataset runs) and "
        f"{table_path}"
    )


if __name__ == "__main__":
    main()
