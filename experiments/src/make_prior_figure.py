from __future__ import annotations

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(HERE, "results")
OUT = os.path.join(HERE, os.pardir, "paper", "source", "figures")

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
PLUM, GOLD = "#8a5fa8", "#b07820"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#9a9892"
mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8.5, "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.2, "legend.frameon": False,
    "grid.color": "#e4e3df", "grid.linewidth": 0.5,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

STYLE = [("upu", "uPU", ORANGE, "s", "-"), ("nnpu", "nnPU", BLUE, "o", "-"),
         ("cvir", "CVIR", AQUA, "^", "--"), ("dcpu", "DC-PU", PLUM, "D", "--"),
         ("imbpu", "ImbPU", GOLD, "v", ":")]


def pick(df, key):
    return df.loc[df.groupby(key)["crit_leeliu"].idxmax()]


def load():
    old = pd.read_csv(os.path.join(R, "e3_results.csv"))
    old["base"] = old.method.str.rsplit("_x", n=1).str[0]
    old["factor"] = old.method.str.rsplit("_x", n=1).str[1].astype(float)
    old = pick(old, ["dataset", "seed", "method"])
    new = pick(pd.read_csv(os.path.join(R, "gap_prior_results.csv")),
               ["dataset", "seed", "method"])
    cols = ["base", "factor", "bacc"]
    return pd.concat([old[cols], new[cols]], ignore_index=True)


def main():
    d = load()
    piv = d.pivot_table(index="base", columns="factor", values="bacc", aggfunc="mean")
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.5), width_ratios=[1.3, 1])
    fig.subplots_adjust(wspace=0.34)

    ax = axes[0]
    for base, lab, color, mk, ls in STYLE:
        g = piv.loc[base]
        ax.plot(g.index, g.values, color=color, marker=mk, ls=ls, lw=1.5,
                ms=4, mec="white", mew=0.7, zorder=3, label=lab)
    ax.axvline(1.0, color=MUTED, lw=0.6, ls=":", zorder=1)
    ax.annotate("true prior", (1.0, ax.get_ylim()[0]), color=MUTED, fontsize=6.8,
                xytext=(3, 4), textcoords="offset points")
    ax.set_xlabel(r"prior supplied to the method, $k\alpha$")
    ax.set_ylabel("test balanced accuracy")
    ax.set_xticks([0.5, 0.75, 1.0, 1.25, 1.5])
    ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    ax.legend(loc="lower right", handlelength=2.0, borderpad=0.25, labelspacing=0.3)

    ax = axes[1]
    xs = range(len(STYLE))
    lo = [piv.loc[b][0.5] - piv.loc[b][1.0] for b, *_ in STYLE]
    hi = [piv.loc[b][1.5] - piv.loc[b][1.0] for b, *_ in STYLE]
    ax.bar([x - 0.19 for x in xs], lo, 0.36, color="#c0504d", label=r"$0.5\alpha$")
    ax.bar([x + 0.19 for x in xs], hi, 0.36, color="#4f81bd", label=r"$1.5\alpha$")
    ax.axhline(0, color=INK2, lw=0.7)
    ax.set_xticks(list(xs)); ax.set_xticklabels([s[1] for s in STYLE], rotation=20)
    ax.set_ylabel("change from true prior", labelpad=2)
    ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    ax.legend(loc="lower left", handlelength=1.4, borderpad=0.25,
              bbox_to_anchor=(0.0, 0.02), ncol=2, columnspacing=1.0)

    for a in axes:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "fig3_prior.pdf"))
    plt.close(fig)
    print("fig3_prior.pdf rebuilt with five objectives")


if __name__ == "__main__":
    main()
