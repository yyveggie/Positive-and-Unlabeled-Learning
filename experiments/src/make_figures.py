from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(HERE, "results")
OUT = os.path.join(HERE, os.pardir, "paper", "source", "figures")
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#9a9892"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6,
    "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "legend.frameon": False,
    "grid.color": "#e4e3df", "grid.linewidth": 0.5,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

LABEL = {
    "bbe_cvir": r"BBE$\rightarrow$CVIR", "imbpu": "ImbPU", "cvir": "CVIR",
    "bbe_nnpu": r"BBE$\rightarrow$nnPU", "wce": "Weighted CE",
    "distpu_off": "Dist-PU", "nnpu": "nnPU", "vpu_off": "VPU",
    "holistic_off": "HolisticPU",
    "en": "Elkan–Noto", "upu": "uPU", "uasn": r"$U$-as-negative",
}
CORE = ["vpu", "distpu"]


def load_tabular():
    key = ["dataset", "c", "seed", "method"]
    grid = pd.concat([pd.read_csv(os.path.join(R, f)) for f in
                      ["p1_results.csv", "p15_results.csv", "ext_results.csv"]],
                     ignore_index=True)
    grid = grid[~grid.method.isin(["tice_nnpu"] + CORE)]
    sel = grid.loc[grid.groupby(key)["crit_leeliu"].idxmax()]
    off = pd.read_csv(os.path.join(R, "officialgrid_results.csv"))
    off = off[off.phase == "tab"]
    off = off.loc[off.groupby(key)["crit_leeliu"].idxmax()]
    return pd.concat([sel, off], ignore_index=True)


def fig_cd():
    tab = load_tabular()
    blocks = tab.pivot_table(index=["dataset", "c", "seed"], columns="method",
                             values="bacc").dropna()
    ranks = blocks.rank(axis=1, ascending=False).mean().sort_values()
    k, N = len(ranks), len(blocks)
    q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
           8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268}[k]
    cd = q05 * np.sqrt(k * (k + 1) / (6.0 * N))

    lo, hi = ranks.min() - 0.35, ranks.max() + 0.35
    half = (k + 1) // 2
    row_h, first_row = 0.115, -0.20
    pad = 3.9

    fig, ax = plt.subplots(figsize=(5.4, 2.55))
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(first_row - half * row_h - 0.16, 0.40)
    ax.axis("off")


    ax.plot([lo, hi], [0, 0], color=INK, lw=0.9, solid_capstyle="butt")
    for t in np.arange(np.ceil(lo), np.floor(hi) + 0.01, 1.0):
        ax.plot([t, t], [0, 0.045], color=INK, lw=0.8)
        ax.text(t, 0.065, f"{t:.0f}", ha="center", va="bottom",
                color=INK2, fontsize=7.5)
    ax.text((lo + hi) / 2, 0.30, "average rank (lower is better)", ha="center",
            va="bottom", color=INK2, fontsize=7.5)


    cliques, i = [], 0
    while i < k:
        j = i
        while j + 1 < k and ranks.iloc[j + 1] - ranks.iloc[i] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
        i += 1
    for depth, (a, b) in enumerate(cliques):
        y = -0.042 - depth * 0.032
        ax.plot([ranks.iloc[a] - 0.04, ranks.iloc[b] + 0.04], [y, y],
                color=ORANGE, lw=2.4, solid_capstyle="round", zorder=2)


    for i, m in enumerate(ranks.index):
        r = ranks[m]
        left = i < half
        row = i if left else (k - 1 - i)
        y = first_row - row * row_h
        edge = lo - 0.12 if left else hi + 0.12
        ax.plot([r, r], [0, y], color=INK, lw=0.7, zorder=1)
        ax.plot([r, edge], [y, y], color=INK, lw=0.7, zorder=1)
        txt = f"{LABEL[m]}  ({r:.2f})" if left else f"({r:.2f})  {LABEL[m]}"
        ax.text(edge - 0.08 if left else edge + 0.08, y, txt,
                ha="right" if left else "left", va="center",
                color=INK, fontsize=7.5)


    ybar = 0.175
    ax.plot([lo, lo + cd], [ybar, ybar], color=BLUE, lw=1.6, solid_capstyle="butt")
    for x in (lo, lo + cd):
        ax.plot([x, x], [ybar - 0.028, ybar + 0.028], color=BLUE, lw=1.0)
    ax.text(lo + cd / 2, ybar + 0.038, f"CD = {cd:.2f}", ha="center", va="bottom",
            color=BLUE, fontsize=7.5)

    fig.savefig(os.path.join(OUT, "fig1_cd.pdf"))
    plt.close(fig)
    print(f"fig1_cd.pdf   k={k} N={N} CD={cd:.3f} cliques={len(cliques)}")


def fig_oracle():
    key = ["dataset", "c", "seed", "method"]

    def gap(df, restrict_c=None):
        if restrict_c is not None:
            df = df[df.c.isin(restrict_c)]
        pu = df.loc[df.groupby(key)["crit_leeliu"].idxmax()]
        orc = df.loc[df.groupby(key)["crit_oraclef1"].idxmax()]
        m = orc.merge(pu, on=key, suffixes=("_or", "_pu"))
        m["g"] = m.f1_or - m.f1_pu
        return m[key + ["g"]]

    def clustered_summary(df, n_boot=10000):
        datasets = sorted(df.dataset.unique())
        cs = sorted(df.c.unique())
        seeds = sorted(df.seed.unique())
        methods = sorted(df.method.unique())
        idx = pd.MultiIndex.from_product(
            [datasets, cs, seeds], names=["dataset", "c", "seed"])
        p = df.pivot(index=["dataset", "c", "seed"], columns="method",
                     values="g").reindex(index=idx, columns=methods)
        if p.isna().any().any():
            raise ValueError("incomplete oracle-gap grid")
        x = p.to_numpy().reshape(len(datasets), len(cs), len(seeds), len(methods))
        rng = np.random.default_rng(20260815)
        boot = np.empty((n_boot, len(methods)))
        for b in range(n_boot):
            total = np.zeros(len(methods))
            for d in rng.integers(0, len(datasets), size=len(datasets)):
                for c in rng.integers(0, len(cs), size=len(cs)):
                    ss = rng.integers(0, len(seeds), size=len(seeds))
                    total += x[d, c, ss].mean(axis=0)
            boot[b] = total / (len(datasets) * len(cs))
        out = pd.DataFrame(index=methods)
        out["mean"] = x.mean(axis=(0, 1, 2))
        out["lo"], out["hi"] = np.quantile(boot, [0.025, 0.975], axis=0)
        return out

    p1 = pd.read_csv(os.path.join(R, "p1_results.csv"))
    p2 = pd.read_csv(os.path.join(R, "p2_results.csv"))
    small = clustered_summary(
        gap(p1[p1.method != "tice_nnpu"], restrict_c=[0.3, 0.5]))
    large = clustered_summary(gap(p2))
    common = [m for m in large.index if m in small.index]
    common.sort(key=lambda m: large.loc[m, "mean"])

    y = np.arange(len(common)); h = 0.36
    fig, ax = plt.subplots(figsize=(4.0, 2.4))
    ax.barh(y + h / 2, small.loc[common, "mean"], height=h, color=BLUE,
            xerr=np.vstack([small.loc[common, "mean"] - small.loc[common, "lo"],
                            small.loc[common, "hi"] - small.loc[common, "mean"]]),
            error_kw=dict(lw=0.7, ecolor=MUTED),
            label="4 configurations", zorder=3)
    ax.barh(y - h / 2, large.loc[common, "mean"], height=h, color=ORANGE,
            xerr=np.vstack([large.loc[common, "mean"] - large.loc[common, "lo"],
                            large.loc[common, "hi"] - large.loc[common, "mean"]]),
            error_kw=dict(lw=0.7, ecolor=MUTED),
            label="24 configurations", zorder=3)
    ax.set_yticks(y, [LABEL[m] for m in common])
    ax.set_xlabel(r"oracle-selection gap in test $F_1$")
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.xaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2,
              title="hyperparameter search space", title_fontsize=7.5,
              columnspacing=1.6, handlelength=1.4)
    fig.savefig(os.path.join(OUT, "fig2_oracle.pdf"))
    plt.close(fig)
    print("fig2_oracle.pdf  " + ", ".join(
        f"{m}:{small.loc[m,'mean']:.3f}->{large.loc[m,'mean']:.3f}" for m in common))


def fig_prior():
    e3 = pd.read_csv(os.path.join(R, "e3_results.csv"))
    e3["base"] = e3.method.str.split("_x").str[0]
    e3["k"] = e3.method.str.split("_x").str[1].astype(float)
    sel = e3.loc[e3.groupby(["dataset", "seed", "method"])["crit_leeliu"].idxmax()]

    tab = load_tabular()
    ref = tab[tab.c == 0.5]
    panels = [("mean over the five datasets", None), ("Bank ($\\alpha=0.12$)", "bank")]

    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.4), sharey=True)
    handles = None
    for pi, (ax, (title, ds)) in enumerate(zip(axes, panels)):
        s = sel if ds is None else sel[sel.dataset == ds]
        r = ref if ds is None else ref[ref.dataset == ds]
        hs = []
        for base, color, mk in [("nnpu", BLUE, "o"), ("upu", ORANGE, "s")]:
            g = s[s.base == base].groupby("k")["bacc"].mean()
            (h,) = ax.plot(g.index, g.values, color=color, marker=mk, ls="-",
                           lw=1.6, ms=4.5, mec="white", mew=0.8, zorder=3,
                           label=f"{LABEL[base]} (needs $\\alpha$)")
            hs.append(h)
            if pi == 1:
                ax.annotate(LABEL[base], (g.index[-1], g.values[-1]), color=color,
                            fontsize=7.5, xytext=(4, 0),
                            textcoords="offset points", va="center")
        for meth, color, ls in [("wce", AQUA, (0, (4, 2))),
                                ("vpu_off", MUTED, (0, (1.5, 1.8)))]:
            v = r[r.method == meth]["bacc"].mean()
            h = ax.axhline(v, color=color, ls=ls, lw=1.3, zorder=2,
                           label=f"{LABEL[meth]} (prior-free)")
            hs.append(h)
        ax.axvline(1.0, color=MUTED, lw=0.6, ls=":", zorder=1)
        ax.set_title(title, color=INK)
        ax.set_xlabel(r"prior fed to the method, $k\alpha$")
        ax.set_xticks([0.5, 0.75, 1.0, 1.25, 1.5])
        ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        handles = handles or hs
    axes[0].set_ylabel("test balanced accuracy")
    axes[0].legend(handles=handles, loc="lower right", fontsize=7,
                   handlelength=2.2, borderpad=0.3, labelspacing=0.35)
    axes[1].annotate("true prior", (1.0, axes[1].get_ylim()[0]), color=MUTED,
                     fontsize=6.8, xytext=(3, 4), textcoords="offset points")
    fig.savefig(os.path.join(OUT, "fig3_prior.pdf"))
    plt.close(fig)
    print("fig3_prior.pdf  panels: mean, bank (prior-free refs: weighted CE, VPU)")


def fig_mechanism():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from sklearn.linear_model import LogisticRegression
    from pu_lib import driving_feature, load_dataset, make_pu

    X, y = load_dataset("spambase", seed=0)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    pos = np.flatnonzero(y == 1)
    drivers = {"attribute": Z[pos, driving_feature(Z, y, "max")],
               "posterior": LogisticRegression(max_iter=1000)
               .fit(Z, y).predict_proba(Z)[pos, 1]}

    def curve(mech, driver, bins=8, seeds=range(10)):
        q = np.quantile(driver, np.linspace(0, 1, bins + 1))
        q[-1] += 1e-9
        idx = np.clip(np.digitize(driver, q[1:-1]), 0, bins - 1)
        mid = [driver[idx == b].mean() for b in range(bins)]
        frac = np.zeros(bins)
        for s in seeds:
            lab = make_pu(Z, y, 0.3, mech, s).s[pos].astype(bool)
            frac += [lab[idx == b].mean() for b in range(bins)]
        return mid, frac / len(seeds)

    panels = [("attribute", "attribute most correlated with the class",
               [("scar", "SCAR", MUTED, "o", (0, (1.5, 1.8))),
                ("sar_attr", "SAR, attribute", BLUE, "s", "-"),
                ("sar_extreme", "SAR, severe", ORANGE, "^", (0, (4, 2)))]),
              ("posterior", r"oracle posterior $\Pr(y{=}1 \mid x)$",
               [("scar", "SCAR", MUTED, "o", (0, (1.5, 1.8))),
                ("sar_gap", "SAR, probabilistic gap", AQUA, "D", "-")])]

    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.3), sharey=True)
    for ax, (dk, xlab, series) in zip(axes, panels):
        for mech, name, col, mk, ls in series:
            mid, frac = curve(mech, drivers[dk])
            ax.plot(mid, frac, color=col, marker=mk, ls=ls, lw=1.5, ms=4,
                    mec="white", mew=0.7, zorder=3, label=name)
        ax.axhline(0.3, color=INK2, lw=0.5, ls=":", zorder=1)
        ax.set_xlabel(xlab)
        ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
        ax.legend(loc="upper left", fontsize=6.8, handlelength=1.9,
                  borderpad=0.3, labelspacing=0.3)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel(r"fraction of positives labeled, $e(x)$")
    axes[0].annotate(r"target $c = 0.3$", (axes[0].get_xlim()[0], 0.3),
                     color=INK2, fontsize=6.5, xytext=(3, -9),
                     textcoords="offset points")
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT, "fig4_mechanism.pdf"))
    plt.close(fig)
    print("fig4_mechanism.pdf  realized propensity on spambase, c = 0.3")


def fig_calib():
    d = pd.read_csv(os.path.join(R, "calib_results.csv"))
    sel = d.loc[d.groupby(["dataset", "mechanism", "c", "seed",
                           "method"])["crit_leeliu"].idxmax()]
    M = ["bacc", "f1", "auc", "auprc", "f1_bestthr", "ece", "brier", "thr_best"]
    g = sel.pivot_table(index="mechanism", values=M, aggfunc="mean")
    order = ["sar_gap", "sar_attr", "sar_extreme"]
    lab = {"sar_gap": "gap", "sar_attr": "attribute", "sar_extreme": "severe"}
    base = g.loc["scar"]
    groups = [("thresholded", ["bacc", "f1"], BLUE, "o", "-"),
              ("ranking", ["auc", "auprc", "f1_bestthr"], AQUA, "s", (0, (4, 2))),
              ("calibration", ["ece", "brier"], ORANGE, "^", (0, (1.5, 1.8)))]

    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.4))
    ax = axes[0]
    x = np.arange(len(order))
    for name, cols, col, mk, ls in groups:
        v = [100 * ((g.loc[m, cols] - base[cols]) / base[cols].abs()).mean()
             for m in order]
        if name == "calibration":
            v = [-u for u in v]
        ax.plot(x, v, color=col, marker=mk, ls=ls, lw=1.6, ms=4.5, mec="white",
                mew=0.8, zorder=3, label=name)
        ax.annotate(name, (x[-1], v[-1]), color=col, fontsize=7.5,
                    xytext=(4, -1), textcoords="offset points", va="center")
    ax.axhline(0, color=MUTED, lw=0.6, zorder=1)
    ax.set_xticks(x, [lab[m] for m in order])
    ax.set_xlim(-0.25, len(order) - 0.25 + 1.15)
    ax.set_xlabel("labeling mechanism, by severity")
    ax.set_ylabel("degradation against SCAR (\\%)")
    ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)

    ax = axes[1]
    xs = np.arange(len(order) + 1)
    mm = ["scar"] + order
    f1 = [g.loc[m, "f1"] for m in mm]
    fb = [g.loc[m, "f1_bestthr"] for m in mm]
    ax.fill_between(xs, f1, fb, color=BLUE, alpha=0.10, zorder=1)
    ax.plot(xs, fb, color=BLUE, marker="s", ls=(0, (4, 2)), lw=1.5, ms=4,
            mec="white", mew=0.8, zorder=3)
    ax.plot(xs, f1, color=BLUE, marker="o", ls="-", lw=1.6, ms=4.5,
            mec="white", mew=0.8, zorder=3)
    ax.annotate("at the best threshold", (xs[0], fb[0]), color=BLUE, fontsize=7,
                xytext=(2, 5), textcoords="offset points")
    ax.annotate("at $0.5$", (xs[0], f1[0]), color=BLUE, fontsize=7,
                xytext=(2, -10), textcoords="offset points")
    ax.annotate("recoverable\nby re-thresholding", (xs[-1], (f1[-1] + fb[-1]) / 2),
                color=INK2, fontsize=6.8, ha="right",
                xytext=(-6, 0), textcoords="offset points", va="center")
    ax.set_xticks(xs, ["SCAR"] + [lab[m] for m in order])
    ax.set_xlabel("labeling mechanism, by severity")
    ax.set_ylabel(r"test $F_1$")
    ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    for a in axes:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT, "fig5_calib.pdf"))
    plt.close(fig)
    print("fig5_calib.pdf  metric families and the threshold headroom")


def fig_graph():
    d = pd.read_csv(os.path.join(R, "graph_results.csv"))
    sel = d.loc[d.groupby(["dataset", "rewire", "backbone", "c", "seed",
                           "method"])["crit_leeliu"].idxmax()]
    h = sel.groupby("rewire").homophily.mean()
    rew = sorted(sel.rewire.unique())
    hx = [h[r] for r in rew]
    COR, NAI = ["nnpu", "imbpu", "cvir"], ["uasn", "wce"]

    def m(bb, meths, r=None):
        s = sel[(sel.backbone == bb) & (sel.method.isin(meths))]
        return (s[s.rewire == r] if r is not None else s).bacc.mean()

    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.4))
    ax = axes[0]
    for name, meths, col, mk in [("risk-corrected", COR, BLUE, "o"),
                                 ("naive", NAI, ORANGE, "s")]:
        ax.plot(hx, [m("gcn", meths, r) for r in rew], color=col, marker=mk,
                lw=1.6, ms=4.5, mec="white", mew=0.8, zorder=3, label=name)
        ax.axhline(m("mlp", meths), color=col, ls=(0, (1.5, 1.8)), lw=1.2,
                   zorder=2)
        ax.annotate(f"{name}, MLP", (hx[-1], m("mlp", meths)), color=col,
                    fontsize=6.8, xytext=(2, 4), textcoords="offset points",
                    ha="right")
    ax.set_xlabel("edge homophily after rewiring")
    ax.set_ylabel("test balanced accuracy")
    ax.set_title("GCN (solid) against MLP (dotted)", color=INK, pad=3)
    ax.invert_xaxis()
    ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=7, handlelength=1.8, borderpad=0.3)

    ax = axes[1]
    b = sel[sel.method == "bbe_nnpu"].copy()
    b["re"] = (b.alpha_used - b.alpha_true).abs() / b.alpha_true
    e = b.pivot_table(index="rewire", columns="backbone", values="re",
                      aggfunc="mean")
    ax.plot(hx, [e.loc[r, "gcn"] for r in rew], color=BLUE, marker="o", lw=1.6,
            ms=4.5, mec="white", mew=0.8, zorder=3)
    ax.axhline(e["mlp"].mean(), color=MUTED, ls=(0, (1.5, 1.8)), lw=1.3, zorder=2)
    ax.annotate("MLP", (hx[0], e["mlp"].mean()), color=INK2, fontsize=7,
                xytext=(-2, 4), textcoords="offset points")
    ax.annotate("GCN", (hx[-1], e.loc[rew[-1], "gcn"]), color=BLUE, fontsize=7,
                xytext=(4, -2), textcoords="offset points")
    ax.set_xlabel("edge homophily after rewiring")
    ax.set_ylabel(r"relative error of $\hat{\alpha}$")
    ax.set_title("class-prior estimation", color=INK, pad=3)
    ax.invert_xaxis()
    ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    for a in axes:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT, "fig6_graph.pdf"))
    plt.close(fig)
    print("fig6_graph.pdf  homophily sweep, accuracy and prior error")


def fig_rare():
    d = pd.read_csv(os.path.join(R, "e5_results.csv"))
    key = ["setting", "dataset", "alpha_target", "c", "seed", "method"]
    sel = d.loc[d.groupby(key, dropna=False)["crit_leeliu"].idxmax()]
    FREE = {"uasn", "wce", "vpu_off", "holistic_off"}
    sel = sel.assign(grp=["prior-free" if m in FREE else "needs $\\alpha$"
                          for m in sel.method])
    sw, fr = sel[sel.setting == "sweep"], sel[sel.setting == "fraud"]
    al = sorted(sw.alpha_target.unique(), reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.4), sharex=True)
    for ax, metric, name in zip(axes, ["f1", "auprc"],
                                [r"test $F_1$", "average precision"]):
        for grp, col, mk, ls in [("prior-free", AQUA, "o", "-"),
                                 ("needs $\\alpha$", BLUE, "s", (0, (4, 2)))]:
            g = sw[sw.grp == grp].groupby("alpha_target")[metric].mean()
            ax.plot(al, [g[a] for a in al], color=col, marker=mk, ls=ls, lw=1.6,
                    ms=4.5, mec="white", mew=0.8, zorder=3, label=grp)
            v = fr[fr.grp == grp][metric].mean()
            ax.plot([0.006], [v], color=col, marker=mk, ms=4.5, mec="white",
                    mew=0.8, zorder=3)
        ax.axvspan(0.0042, 0.0086, color="#eceae5", zorder=0)
        ax.set_xscale("log")
        ax.set_xticks([0.10, 0.05, 0.02, 0.006],
                      ["0.10", "0.05", "0.02", "0.0017"])
        ax.set_xlabel(r"class prior $\alpha$")
        ax.set_ylabel(name)
        ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(loc="lower right", fontsize=7, handlelength=1.8,
                   borderpad=0.3)
    for ax in axes:
        ax.annotate("fraud", (0.006, ax.get_ylim()[0]), color=INK2, fontsize=6.5,
                    ha="center", va="bottom", xytext=(0, 2),
                    textcoords="offset points")
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT, "fig7_rare.pdf"))
    plt.close(fig)
    print("fig7_rare.pdf  family means against the class prior")


if __name__ == "__main__":
    fig_cd()
    fig_oracle()
    fig_prior()
    fig_mechanism()
    fig_calib()
    fig_graph()
    fig_rare()
    print(f"\n-> {OUT}")
