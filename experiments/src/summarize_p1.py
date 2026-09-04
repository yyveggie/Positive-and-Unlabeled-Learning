from __future__ import annotations

import os, sys
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "p1_results.csv")

df = pd.read_csv(CSV)
print(f"{len(df)} rows, datasets={sorted(df.dataset.unique())}, "
      f"c={sorted(df.c.unique())}, seeds={sorted(df.seed.unique())}\n")


key = ["dataset", "c", "seed", "method"]
pu_sel = df.loc[df.groupby(key)["crit_leeliu"].idxmax()].copy()
or_sel = df.loc[df.groupby(key)["crit_oraclef1"].idxmax()].copy()
pu_sel["selection"], or_sel["selection"] = "pu_only", "oracle"


print("=" * 70)
print("E1  test balanced accuracy, PU-only model selection (mean over c & seeds)")
e1 = pu_sel.pivot_table(index="method", columns="dataset", values="bacc", aggfunc="mean")
e1["MEAN"] = e1.mean(axis=1)
print(e1.round(3).sort_values("MEAN", ascending=False).to_string())


blocks = pu_sel.pivot_table(index=["dataset", "c", "seed"], columns="method", values="bacc")
blocks = blocks.dropna()
ranks = blocks.rank(axis=1, ascending=False)
print("\naverage rank (lower = better):")
print(ranks.mean().round(2).sort_values().to_string())
if blocks.shape[1] >= 3 and len(blocks) >= 5:
    stat, p = friedmanchisquare(*[blocks[m] for m in blocks.columns])
    print(f"\nFriedman test: chi2={stat:.1f}, p={p:.2e}")


print("\nbacc by label frequency c (rows: method):")
print(pu_sel.pivot_table(index="method", columns="c", values="bacc",
                         aggfunc="mean").round(3).to_string())


print("\n" + "=" * 70)
print("E2  oracle-selection gap  (test F1 with oracle hp  minus  PU-only hp)")
merged = or_sel.merge(pu_sel, on=key, suffixes=("_or", "_pu"))
merged["gap_f1"] = merged["f1_or"] - merged["f1_pu"]
merged["gap_bacc"] = merged["bacc_or"] - merged["bacc_pu"]
gap = merged.groupby("method")[["gap_f1", "gap_bacc"]].agg(["mean", "std"])
print(gap.round(4).to_string())
print("\npositive gap = the method benefited from labels it should not have had")


tice = df[df.method == "tice_nnpu"].drop_duplicates(subset=["dataset", "c", "seed"])
if len(tice):
    tice = tice.assign(rel_err=(tice.alpha_used - tice.alpha_true).abs() / tice.alpha_true)
    print("\n" + "=" * 70)
    print("TIcE prior estimation relative error |a_hat - a|/a, by dataset:")
    print(tice.groupby("dataset")["rel_err"].agg(["mean", "std"]).round(3).to_string())

out = os.path.join(HERE, "results", "summary_e1_e2.txt")
print(f"\n(rerun with stdout redirect to save: > {out})")
