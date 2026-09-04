from __future__ import annotations

import os
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = pd.read_csv(os.path.join(HERE, "results", "graph_results.csv"))
KEY = ["dataset", "rewire", "backbone", "c", "seed", "method"]
sel = d.loc[d.groupby(KEY)["crit_leeliu"].idxmax()]
ORDER = ["uasn", "wce", "nnpu", "imbpu", "cvir", "bbe_nnpu"]
NAIVE, CORRECTED = ["uasn", "wce"], ["nnpu", "imbpu", "cvir"]

h = sel.groupby(["dataset", "rewire"]).homophily.mean().round(3)
print(f"{len(d)} runs, {len(sel)} after PU-only selection")
print("edge homophily by rewiring level:")
print(h.to_string(), "\n")

print("=" * 78)
print("A. Balanced accuracy, MLP against GCN (mean over datasets, c, seeds)")
t = sel.pivot_table(index="method", columns=["rewire", "backbone"],
                    values="bacc", aggfunc="mean").reindex(ORDER)
print(t.round(3).to_string())

print("\n" + "=" * 78)
print("B. What propagation is worth, per method (GCN minus MLP)")
g = sel.pivot_table(index=["method", "rewire"], columns="backbone",
                    values="bacc", aggfunc="mean")
g["gain"] = g.gcn - g.mlp
gain = g.gain.unstack().reindex(ORDER)
gain["family"] = ["naive" if m in NAIVE else
                  ("corrected" if m in CORRECTED else "estimated prior")
                  for m in gain.index]
print(gain.round(3).to_string())
print("\nby family:")
print(gain.groupby("family").mean(numeric_only=True).round(3).to_string())
print("\nIf message passing were doing the work of the correction, the naive")
print("objectives would gain most; a gap that survives propagation means it is")
print("not a substitute.")

print("\n" + "=" * 78)
print("C. Class-prior estimation under each backbone")
b = sel[sel.method == "bbe_nnpu"].copy()
b["rel_err"] = (b.alpha_used - b.alpha_true).abs() / b.alpha_true
b["ratio"] = b.alpha_used / b.alpha_true
p = b.pivot_table(index=["dataset", "rewire"], columns="backbone",
                  values=["ratio", "rel_err"], aggfunc="mean")
print(p.round(2).to_string())
print("\nrelative error averaged over datasets:")
print(b.pivot_table(index="rewire", columns="backbone", values="rel_err",
                    aggfunc="mean").round(2).to_string())

print("\n" + "=" * 78)
print("D. Does the ranking survive the change of backbone?")
for bb in ["mlp", "gcn"]:
    r = sel[sel.backbone == bb].groupby("method").bacc.mean().reindex(ORDER)
    print(f"  {bb}: " + ", ".join(f"{m} {v:.3f}" for m, v in
                                  r.sort_values(ascending=False).items()))

print("\n" + "=" * 78)
print("E. Scarce labeling on graphs (c = 0.1 against c = 0.5, GCN)")
q = sel[sel.backbone == "gcn"].pivot_table(index="method", columns="c",
                                           values="bacc", aggfunc="mean")
print(q.reindex(ORDER).round(3).to_string())
