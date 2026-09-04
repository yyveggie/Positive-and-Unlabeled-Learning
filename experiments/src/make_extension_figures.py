from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT.parent / "paper" / "source" / "figures"
FIGURES.mkdir(exist_ok=True)

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
PURPLE, GOLD, GRAY = "#8658a5", "#c28b16", "#666666"
PALETTE = [BLUE, ORANGE, GREEN, PURPLE, GOLD, GRAY]
MARKERS = ["o", "s", "^", "D", "P", "X"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#9a9892"

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8.5, "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7, "legend.frameon": False,
    "grid.color": "#e4e3df", "grid.linewidth": 0.5,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})


def finish(fig, name):
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png", dpi=240)
    plt.close(fig)


def prior_estimators():
    data = pd.read_csv(RESULTS / "prior_estimation_track.csv")
    order = ["BBE", "DEDPUL", "KM1", "KM2", "TIcE"]
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.25))
    for index, estimator in enumerate(order):
        part = data[data.estimator == estimator]
        g = part.groupby("separation").abs_error.agg(["mean", "sem"])
        axes[0].errorbar(
            g.index, g["mean"], yerr=1.96 * g["sem"], label=estimator,
            color=PALETTE[index], marker=MARKERS[index], lw=1.35, ms=3.8,
            capsize=2,
        )
        g = part.groupby("alpha_true").signed_error.agg(["mean", "sem"])
        axes[1].errorbar(
            100 * g.index, g["mean"], yerr=1.96 * g["sem"], label=estimator,
            color=PALETTE[index], marker=MARKERS[index], lw=1.35, ms=3.8,
            capsize=2,
        )
    axes[0].set(xlabel="Gaussian separation", ylabel="mean absolute prior error",
                title="(a) Error falls with separation")
    axes[0].set_xticks([0.5, 1, 2, 4]); axes[0].grid(axis="y")
    axes[1].axhline(0, color=MUTED, lw=0.7)
    axes[1].set(xlabel="true positive prior (%)", ylabel="mean signed error",
                title="(b) Positive rare-prior bias on average")
    axes[1].set_xticks([2, 5, 10, 30]); axes[1].grid(axis="y")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.03))
    fig.subplots_adjust(bottom=0.25, wspace=0.34)
    finish(fig, "fig8_prior_estimators")


def rare_dense():
    data = pd.read_csv(RESULTS / "rare_dense_track.csv")
    order = ["uasn", "wce", "nnpu", "imbpu", "cvir", "bbe_cvir"]
    label = {
        "uasn": r"$U$-as-negative", "wce": "weighted CE", "nnpu": "nnPU",
        "imbpu": "ImbPU", "cvir": "CVIR", "bbe_cvir": r"BBE$\to$CVIR",
    }
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.75),
                             gridspec_kw={"width_ratios": [1.5, 1]})
    for index, method in enumerate(order):
        g = data[data.method == method].groupby("alpha_target").agg(
            mean=("normalized_ap_lift", "mean"),
            sem=("normalized_ap_lift", "sem"),
        ).sort_index()
        axes[0].errorbar(
            100 * g.index, g["mean"], yerr=1.96 * g["sem"], label=label[method],
            color=PALETTE[index], marker=MARKERS[index], lw=1.3, ms=3.6,
            capsize=1.8,
        )
    axes[0].axvspan(3, 5, color=ORANGE, alpha=0.08, lw=0)
    axes[0].set(xlabel="target positive prior (%)",
                ylabel="normalized AP lift above prevalence",
                title="(a) No shared cutoff in tested curves")
    axes[0].set_xticks([1, 2, 3, 5, 7.5, 10]); axes[0].grid(axis="y")

    bbe = pd.read_csv(RESULTS / "rare_dense_bbe_summary.csv").sort_values(
        "alpha_target"
    )
    axes[1].plot(100 * bbe.alpha_target, bbe.mean_ratio, color=BLUE,
                 marker="o", lw=1.5, ms=4)
    axes[1].axhline(1, color=MUTED, lw=0.8, ls="--")
    axes[1].axvspan(3, 5, color=ORANGE, alpha=0.08, lw=0)
    axes[1].set_yscale("log")
    axes[1].set(xlabel="target positive prior (%)",
                ylabel=r"BBE estimate / true $\alpha$",
                title="(b) BBE error grows smoothly")
    axes[1].set_xticks([1, 2, 3, 5, 7.5, 10],
                       ["1", "2", "3", "5", "7.5", "10"])
    axes[1].grid(axis="y")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(bottom=0.33, wspace=0.42)
    finish(fig, "fig9_rare_dense")


def beyond_and_vision():
    data = pd.read_csv(RESULTS / "beyond_scar_combined.csv")
    sar = data[data.mechanism != "scar"]
    order = ["neg", "scar_c", "sar_em_all", "sar_em_oracle", "pg_ipw"]
    label = {
        "neg": r"$U$-as-negative", "scar_c": "SCAR (known c)",
        "sar_em_all": "SAR-EM (all)", "sar_em_oracle": "SAR-EM (oracle attrs.)",
        "pg_ipw": "PG-IPW",
    }
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.3))
    for index, method in enumerate(order):
        part = sar[sar.method_id == method]
        axes[0].scatter(part.auprc.mean(), part.bacc.mean(), s=31,
                        marker=MARKERS[index], color=PALETTE[index],
                        label=label[method], zorder=3)
    axes[0].set(xlabel="average precision", ylabel="balanced accuracy",
                title="(a) Beyond-SCAR trade-off\nacross tested SAR cells")
    axes[0].grid(); axes[0].legend(loc="lower left", fontsize=6.6)

    vision_path = RESULTS / "end_to_end_vision_track.csv"
    if vision_path.exists():
        vision = pd.read_csv(vision_path)
        methods = ["uasn_e2e", "nnpu_e2e", "holistic_e2e"]
        vlabel = {"uasn_e2e": r"$U$-as-N", "nnpu_e2e": "nnPU",
                  "holistic_e2e": "HolisticPU-style"}
        dataset_marker = {"cifar10": "o", "fashionmnist": "s"}
        for index, method in enumerate(methods):
            part = vision[vision.method == method]
            for dataset in ["cifar10", "fashionmnist"]:
                cell = part[part.dataset == dataset]
                axes[1].scatter(
                    cell.auprc, cell.bacc, s=17, alpha=0.28,
                    color=PALETTE[index], marker=dataset_marker[dataset],
                    linewidths=0, zorder=2,
                )
                axes[1].scatter(
                    cell.auprc.mean(), cell.bacc.mean(), s=49,
                    color=PALETTE[index], marker=dataset_marker[dataset],
                    edgecolor="white", linewidth=0.7, zorder=3,
                )
        method_handles = [
            Line2D([0], [0], marker="o", linestyle="none", markersize=5,
                   markerfacecolor=PALETTE[index], markeredgecolor="none",
                   label=vlabel[method])
            for index, method in enumerate(methods)
        ]
        dataset_handles = [
            Line2D([0], [0], marker=dataset_marker[dataset], linestyle="none",
                   markersize=5, markerfacecolor=GRAY, markeredgecolor="none",
                   label=label)
            for dataset, label in [("cifar10", "CIFAR-10"),
                                   ("fashionmnist", "Fashion-MNIST")]
        ]
        method_legend = axes[1].legend(
            handles=method_handles, loc="upper left", fontsize=6.5,
            handletextpad=0.3,
        )
        axes[1].add_artist(method_legend)
        axes[1].legend(
            handles=dataset_handles, loc="upper left",
            bbox_to_anchor=(0.0, 0.64), fontsize=6.5, handletextpad=0.3,
        )
        axes[1].set(xlabel="average precision", ylabel="balanced accuracy",
                    title="(b) Ranking versus fixed-threshold\nperformance")
        axes[1].grid()
    else:
        axes[1].text(0.5, 0.5, "end-to-end results pending", ha="center",
                     va="center", transform=axes[1].transAxes, color=MUTED)
        axes[1].set_axis_off()
    fig.subplots_adjust(wspace=0.35)
    finish(fig, "fig10_extension_tracks")


def main():
    prior_estimators()
    rare_dense()
    beyond_and_vision()
    print("wrote fig8_prior_estimators, fig9_rare_dense, fig10_extension_tracks")


if __name__ == "__main__":
    main()
