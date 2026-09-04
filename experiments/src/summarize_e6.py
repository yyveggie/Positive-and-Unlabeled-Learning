from __future__ import annotations

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = pd.read_csv(os.path.join(HERE, "results", "e6_results.csv"))
key = ["task", "kind", "c", "seed", "method"]
sel = d.loc[d.groupby(key)["crit_leeliu"].idxmax()]
ORDER = ["uasn", "wce", "en", "upu", "nnpu", "imbpu", "cvir", "bbe_nnpu"]
print(f"{len(d)} runs, {len(sel)} after PU-only selection\n")

for task in sel.task.unique():
    s = sel[sel.task == task]
    print("=" * 70)
    print(f"{task}: balanced accuracy by representation")
    t = s.pivot_table(index="method", columns="kind", values="bacc",
                      aggfunc="mean").reindex(ORDER)
    if {"raw", "pretrained"} <= set(t.columns):
        t = t[["raw", "pretrained"]]
        t["gain"] = t.pretrained - t.raw
    print(t.round(3).to_string())
    if {"raw", "pretrained"} <= set(t.columns):
        spread = t[["raw", "pretrained"]].agg(["max", "min"])
        rng = spread.loc["max"] - spread.loc["min"]
        print(f"\n  spread between best and worst method: "
              f"raw {rng['raw']:.3f} -> pretrained {rng['pretrained']:.3f}")
        gap = (t.loc[["uasn"], ["raw", "pretrained"]].values[0]
               - spread.loc["max"].values)
        print(f"  how far U-as-negative sits behind the best method: "
              f"raw {gap[0]:+.3f} -> pretrained {gap[1]:+.3f}")
    print()

print("=" * 70)
print("Pooled over both tasks: spread and the naive baseline's deficit")
rows = []
for (task, kind), g in sel.groupby(["task", "kind"]):
    m = g.groupby("method")["bacc"].mean()
    rows.append(dict(task=task, kind=kind, best=m.max(), worst=m.min(),
                     spread=m.max() - m.min(), uasn=m.get("uasn", np.nan),
                     uasn_deficit=m.max() - m.get("uasn", np.nan)))
r = pd.DataFrame(rows).sort_values(["task", "kind"])
print(r.round(3).to_string(index=False))
