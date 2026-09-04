from __future__ import annotations

import os
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = pd.read_csv(os.path.join(HERE, "results", "e5_results.csv"))
key = ["setting", "dataset", "alpha_target", "c", "seed", "method"]
sel = d.loc[d.groupby(key, dropna=False)["crit_leeliu"].idxmax()]
ORDER = ["uasn", "wce", "en", "upu", "nnpu", "imbpu", "cvir", "bbe_nnpu",
         "bbe_cvir", "vpu_off", "distpu_off"]
PRIOR_FREE = {"uasn", "wce", "vpu_off"}
print(f"{len(d)} runs, {len(sel)} after PU-only selection")
print(f"methods: {sorted(sel.method.unique())}\n")

sw = sel[sel.setting == "sweep"]
for metric in ["f1", "auprc", "f1_bestthr", "bacc", "recall"]:
    print("=" * 76)
    print(f"A. {metric} against the class prior (controlled sweep, "
          f"mean over datasets, c, seeds)")
    t = sw.pivot_table(index="method", columns="alpha_target", values=metric,
                       aggfunc="mean")
    t = t.reindex(ORDER)
    t.insert(0, "prior-free", [m in PRIOR_FREE for m in t.index])
    print(t.round(3).to_string())
    print()

print("=" * 76)
print("B. How often each method degenerates to predicting a single class")
sw2 = sw.assign(dead=((sw.pred_rate == 0) | (sw.pred_rate == 1)).astype(int))
print(sw2.pivot_table(index="method", columns="alpha_target", values="dead",
                      aggfunc="mean").reindex(ORDER).round(2).to_string())

fr = sel[sel.setting == "fraud"]
if len(fr):
    print("\n" + "=" * 76)
    print(f"C. Credit-card fraud, natural prior alpha = {fr.alpha_true.mean():.5f}")
    cols = ["f1", "auprc", "f1_bestthr", "auc", "bacc", "precision", "recall",
            "pred_rate"]
    print(fr.groupby("method")[cols].mean().reindex(ORDER).round(3).to_string())
    print(f"\nan all-positive predictor would score F1 = "
          f"{2 * fr.alpha_true.mean() / (1 + fr.alpha_true.mean()):.4f}, "
          f"AUPRC = {fr.alpha_true.mean():.4f}")

print("\n" + "=" * 76)
print("D. BBE prior estimate against the truth")
b = sel[sel.method == "bbe_nnpu"].copy()
b["ratio"] = b.alpha_used / b.alpha_true
g = b.groupby(["setting", "alpha_target"], dropna=False).agg(
    alpha_true=("alpha_true", "mean"), alpha_hat=("alpha_used", "mean"),
    ratio=("ratio", "mean"))
print(g.round(4).to_string())

print("\n" + "=" * 76)
print("E. Prior-free against prior-dependent, averaged within each group")
sw3 = sw.assign(group=["prior-free" if m in PRIOR_FREE else "prior-dependent"
                       for m in sw.method])
print(sw3.pivot_table(index="group", columns="alpha_target",
                      values=["f1", "auprc"], aggfunc="mean").round(3).to_string())
if len(fr):
    fr3 = fr.assign(group=["prior-free" if m in PRIOR_FREE else "prior-dependent"
                           for m in fr.method])
    print("\nfraud:")
    print(fr3.groupby("group")[["f1", "auprc", "f1_bestthr"]].mean().round(3).to_string())
