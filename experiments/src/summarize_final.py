from __future__ import annotations

import os
import pandas as pd
from scipy.stats import friedmanchisquare

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(HERE, "results")


def pick(df):
    key = ["dataset", "c", "seed", "method"]
    return df.loc[df.groupby(key)["crit_leeliu"].idxmax()]


frames = []
for f in ["p1_results.csv", "p15_results.csv", "ext_results.csv"]:
    p = os.path.join(R, f)
    if os.path.exists(p):
        frames.append(pd.read_csv(p))
df = pd.concat(frames, ignore_index=True)
df = df[df.method != "tice_nnpu"]
sel = pick(df)

def grid_desc(d):
    cs = ",".join(f"{c:g}" for c in sorted(d.c.unique()))
    return "c in {%s} x %d seeds" % (cs, d.seed.nunique())


print("=" * 74)
print(f"GRAND E1  test bacc, PU-only selection, mean over {grid_desc(sel)}")
tab = sel.pivot_table(index="method", columns="dataset", values="bacc", aggfunc="mean")
tab["MEAN"] = tab.mean(axis=1)
print(tab.round(3).sort_values("MEAN", ascending=False).to_string())

blocks = sel.pivot_table(index=["dataset", "c", "seed"], columns="method",
                         values="bacc").dropna()
if len(blocks) >= 5:
    ranks = blocks.rank(axis=1, ascending=False)
    print("\naverage rank (lower = better):")
    print(ranks.mean().round(2).sort_values().to_string())
    stat, p = friedmanchisquare(*[blocks[m] for m in blocks.columns])
    print(f"\nFriedman: chi2={stat:.1f}, p={p:.2e}  (n_blocks={len(blocks)}, "
          f"k={blocks.shape[1]})")

print("\nbacc by label frequency c:")
print(sel.pivot_table(index="method", columns="c", values="bacc",
                      aggfunc="mean").round(3).to_string())

mn = os.path.join(R, "mnist_results.csv")
if os.path.exists(mn):
    m = pick(pd.read_csv(mn))
    print("\n" + "=" * 74)
    print(f"MNIST-20k even/odd ({grid_desc(m)}, PU-only selection)")
    print(m.pivot_table(index="method", columns="c", values="bacc",
                        aggfunc="mean").round(3)
          .assign(MEAN=lambda t: t.mean(axis=1))
          .sort_values("MEAN", ascending=False).to_string())
