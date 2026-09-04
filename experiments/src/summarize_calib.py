from __future__ import annotations

import os
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = pd.read_csv(os.path.join(HERE, "results", "calib_results.csv"))
key = ["dataset", "mechanism", "c", "seed", "method"]
sel = d.loc[d.groupby(key)["crit_leeliu"].idxmax()]

MECHS = ["scar", "sar_attr", "sar_gap", "sar_extreme"]
ORDER = ["uasn", "wce", "en", "upu", "nnpu", "imbpu", "cvir"]
THRESH = ["bacc", "f1"]
RANK = ["auc", "auprc", "f1_bestthr"]
CALIB = ["ece", "brier"]
ALL = THRESH + RANK + CALIB

print(f"{len(d)} runs, {len(sel)} after PU-only selection "
      f"({sel.dataset.nunique()} datasets x {len(MECHS)} mechanisms x "
      f"{sel.c.nunique()} label frequencies x {sel.seed.nunique()} seeds x "
      f"{sel.method.nunique()} methods)\n")

print("=" * 78)
print("A. Every metric by labeling mechanism (mean over datasets, methods, c, seeds)")
t = sel.pivot_table(index="mechanism", values=ALL, aggfunc="mean").reindex(MECHS)
print(t[ALL].round(4).to_string())
print("\nchange from the SCAR reference (ECE and Brier: positive = worse):")
delta = t[ALL] - t.loc["scar", ALL]
print(delta.drop(index="scar").round(4).to_string())

print("\n" + "=" * 78)
print("B. Relative change, which puts the three groups on one scale (%)")
rel = 100 * (t[ALL] - t.loc["scar", ALL]) / t.loc["scar", ALL].abs()
print(rel.drop(index="scar").round(1).to_string())

print("\n" + "=" * 78)
print("C. Per method under the severe mechanism, against its own SCAR row")
a = sel[sel.mechanism == "scar"].groupby("method")[ALL].mean()
b = sel[sel.mechanism == "sar_extreme"].groupby("method")[ALL].mean()
cmp_ = pd.concat({"scar": a, "severe": b, "delta": b - a}, axis=1)
cols = [(g, m) for m in ["bacc", "auc", "f1_bestthr", "ece"]
        for g in ["scar", "severe", "delta"]]
print(cmp_.reindex(ORDER)[cols].round(3).to_string())

print("\n" + "=" * 78)
print("D. How badly is the 0.5 threshold placed under each mechanism?")
g = sel.groupby("mechanism")[["f1", "f1_bestthr", "thr_best", "pred_rate"]] \
       .mean().reindex(MECHS)
g["headroom"] = g.f1_bestthr - g.f1
g["headroom_vs_scar"] = g.headroom - g.loc["scar", "headroom"]
print(g.round(3).to_string())
print("\nheadroom is the F1 recoverable by moving the threshold alone; thr_best")
print("is where the optimum actually sits, and its drift away from 0.5 is the")
print("calibration damage expressed in the units the practitioner sees.")

print("\n" + "=" * 78)
print("E. Per dataset, severe mechanism: bacc drop against ECE increase")
p = []
for ds in sorted(sel.dataset.unique()):
    s = sel[(sel.dataset == ds) & (sel.mechanism == "scar")]
    v = sel[(sel.dataset == ds) & (sel.mechanism == "sar_extreme")]
    p.append(dict(dataset=ds, bacc_drop=s.bacc.mean() - v.bacc.mean(),
                  auc_drop=s.auc.mean() - v.auc.mean(),
                  ece_rise=v.ece.mean() - s.ece.mean(),
                  brier_rise=v.brier.mean() - s.brier.mean()))
print(pd.DataFrame(p).set_index("dataset").round(3).to_string())
