from __future__ import annotations

import os
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(HERE, "results")
d = pd.read_csv(os.path.join(R, "officialgrid_results.csv"))
DEFAULT = {"vpu_off": "lr3e-05|lam0.03", "distpu_off": "warm_lr0.0005|co_mixup5.0",
           "holistic_off": "lr0.001|warm_epochs10"}
KEY = ["phase", "dataset", "c", "seed", "method"]
COLS = ["breast", "spambase", "mushroom", "adult", "bank"]

d["is_default"] = [r.cfg == DEFAULT[r.method] for r in d.itertuples()]
sel = d.loc[d.groupby(KEY)["crit_leeliu"].idxmax()].assign(budget="matched")
dfl = d[d.is_default].assign(budget="default")
both = pd.concat([sel, dfl], ignore_index=True)
print(f"{len(d)} runs, {d.groupby(KEY).size().max()} configurations per cell\n")

print("=" * 78)
print("A. Balanced accuracy, published defaults against a four-point search")
for phase in ["tab", "mnist"]:
    p = both[both.phase == phase]
    if not len(p):
        continue
    t = p.pivot_table(index=["method", "budget"], columns="dataset",
                      values="bacc", aggfunc="mean")
    if phase == "tab":
        t = t[COLS]
        t["MEAN"] = t.mean(axis=1)
    print(f"\n-- {phase} --")
    print(t.round(3).to_string())

print("\n" + "=" * 78)
print("B. What the extra budget buys, per method")
for phase in ["tab", "mnist"]:
    p = both[both.phase == phase]
    if not len(p):
        continue
    g = p.pivot_table(index="method", columns="budget", values=["bacc", "f1", "auc"])
    for m in ["bacc", "f1", "auc"]:
        g[(m, "delta")] = g[(m, "matched")] - g[(m, "default")]
    print(f"\n-- {phase} --")
    print(g.round(3).to_string())

print("\n" + "=" * 78)
print("C. Which configuration the PU-only criterion ends up choosing")
print(sel.groupby(["method", "cfg"]).size().rename("cells").to_string())


print("\n" + "=" * 78)
print("D. The main comparison recomputed with budget-matched official numbers")
grid = pd.concat([pd.read_csv(os.path.join(R, f)) for f in
                  ["p1_results.csv", "p15_results.csv", "ext_results.csv"]],
                 ignore_index=True)
grid = grid[~grid.method.isin(["tice_nnpu", "vpu", "distpu"])]
gsel = grid.loc[grid.groupby(["dataset", "c", "seed", "method"])["crit_leeliu"].idxmax()]

old = pd.read_csv(os.path.join(R, "official_results.csv"))
mn = pd.read_csv(os.path.join(R, "mnist_results.csv"))
mnsel = mn.loc[mn.groupby(["dataset", "c", "seed", "method"])["crit_leeliu"].idxmax()]
mnsel = mnsel[~mnsel.method.isin(["vpu", "distpu"])]

for label, off in [("published defaults", old), ("budget matched", sel)]:
    tabo = off[off.phase.isin(["tab"])]
    mno = off[off.phase == "mnist"]
    t = pd.concat([gsel, tabo], ignore_index=True).pivot_table(
        index="method", columns="dataset", values="bacc", aggfunc="mean")[COLS]
    t["MEAN"] = t.mean(axis=1)
    t["MNIST"] = pd.concat([mnsel, mno], ignore_index=True) \
        .groupby("method")["bacc"].mean()
    print(f"\n-- official implementations at {label} --")
    print(t.round(3).sort_values("MEAN", ascending=False).to_string())
