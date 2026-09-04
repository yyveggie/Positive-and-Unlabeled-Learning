from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(HERE, "results")
SCAR_METHODS = ["uasn", "wce", "en", "upu", "nnpu"]

t = pd.read_csv(os.path.join(R, "scartest_results.csv"))
print(f"{len(t)} test runs\n")

print("=" * 78)
print("A. Rejection rate at alpha = 0.05  (mechanism 'scar' row IS the type-I error)")
for stat in ["ks", "nb_auc"]:
    sub = t[t.stat == stat]
    tab = sub.pivot_table(index="dataset", columns="mechanism", values="reject",
                          aggfunc="mean")
    tab = tab[["scar", "sar_attr", "sar_gap", "sar_extreme"]]
    tab["P-hat purity"] = sub.groupby("dataset")["purity"].mean()
    print(f"\n-- statistic: {stat} --")
    print(tab.round(2).sort_values("P-hat purity", ascending=False).to_string())

print("\n" + "=" * 78)
print("B. Type-I error against the purity of the approximated positive set")
for stat in ["ks", "nb_auc"]:
    sub = t[(t.stat == stat) & (t.mechanism == "scar")]
    g = sub.groupby("dataset").agg(typeI=("reject", "mean"),
                                   purity=("purity", "mean"))
    r, p = pearsonr(g.purity, g.typeI)
    print(f"  {stat:7s} r = {r:+.3f} (p = {p:.3f}) over {len(g)} datasets")


print("\n" + "=" * 78)
print("C. Test rejection vs. actual degradation of SCAR-based classifiers")
key = ["dataset", "c", "seed", "method"]
grid = pd.concat([pd.read_csv(os.path.join(R, f)) for f in
                  ["p1_results.csv", "ext_results.csv"]], ignore_index=True)
grid = grid[grid.method.isin(SCAR_METHODS) & grid.c.isin([0.3, 0.5])]
ref = grid.loc[grid.groupby(key)["crit_leeliu"].idxmax()]
ref = ref.groupby(["dataset", "c"])["bacc"].mean().rename("bacc_scar")

e4 = pd.read_csv(os.path.join(R, "e4_results.csv"))
e4 = e4[e4.method.isin(SCAR_METHODS)]
sel4 = e4.loc[e4.groupby(["dataset", "mechanism", "c", "seed", "method"])
              ["crit_leeliu"].idxmax()]
deg = sel4.groupby(["dataset", "mechanism", "c"])["bacc"].mean().rename("bacc_sar")
deg = deg.reset_index().merge(ref.reset_index(), on=["dataset", "c"])
deg["drop"] = deg.bacc_scar - deg.bacc_sar

rej = (t[t.stat == "ks"].groupby(["dataset", "mechanism", "c"])["reject"]
       .mean().rename("reject_ks").reset_index())
m = deg.merge(rej, on=["dataset", "mechanism", "c"])

print("\nmean balanced-accuracy drop vs. SCAR reference, by mechanism:")
print(m.pivot_table(index="dataset", columns="mechanism",
                    values="drop", aggfunc="mean").round(3).to_string())
print("\nKS rejection rate on the same cells:")
print(m.pivot_table(index="dataset", columns="mechanism",
                    values="reject_ks", aggfunc="mean").round(2).to_string())

r, p = pearsonr(m.reject_ks, m["drop"])
print(f"\ncorrelation(rejection rate, accuracy drop) = {r:+.3f} (p = {p:.3f}), "
      f"n = {len(m)} cells")

clean = m[m.dataset.isin(["mushroom", "spambase", "breast"])]
if len(clean) > 3:
    r2, p2 = pearsonr(clean.reject_ks, clean["drop"])
    print(f"  restricted to the three datasets where the test holds its size: "
          f"r = {r2:+.3f} (p = {p2:.3f}), n = {len(clean)}")
