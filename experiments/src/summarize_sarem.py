from __future__ import annotations

import os
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = pd.read_csv(os.path.join(HERE, "results", "sarem_results.csv"))
d["label"] = d.method + d.prop_spec.map(lambda s: "" if s == "-" else f" ({s})")
ORDER = ["neg", "scar_c", "sar_em (oracle)", "sar_em (all)"]
MECHS = ["scar", "sar_indep", "sar_attr", "sar_gap", "sar_extreme"]
print(f"{len(d)} runs\n")

print("=" * 74)
print("A. Balanced accuracy by labeling mechanism (mean over datasets, c, seeds)")
t = d.pivot_table(index="label", columns="mechanism", values="bacc", aggfunc="mean")
print(t.reindex(ORDER)[MECHS].round(3).to_string())

print("\n" + "=" * 74)
print("B. SAR-EM (oracle) minus the SCAR-with-known-c baseline, per dataset")
piv = d.pivot_table(index=["dataset", "mechanism"], columns="label",
                    values="bacc", aggfunc="mean").reset_index()
piv["gain_oracle"] = piv["sar_em (oracle)"] - piv["scar_c"]
piv["gain_all"] = piv["sar_em (all)"] - piv["scar_c"]
print(piv.pivot_table(index="dataset", columns="mechanism",
                      values="gain_oracle").reindex(columns=MECHS).round(3).to_string())
print("\nsame, with the propensity model given every attribute:")
print(piv.pivot_table(index="dataset", columns="mechanism",
                      values="gain_all").reindex(columns=MECHS).round(3).to_string())

print("\n" + "=" * 74)
print("C. Does SAR-EM recover the loss that SAR inflicts on the SCAR baseline?")
base = d[d.mechanism == "scar"].groupby(["dataset", "label"])["bacc"].mean()
rows = []
for mech in MECHS[1:]:
    sub = d[d.mechanism == mech].groupby(["dataset", "label"])["bacc"].mean()
    for ds in d.dataset.unique():
        try:
            rows.append(dict(dataset=ds, mechanism=mech,
                             scar_drop=base[(ds, "scar_c")] - sub[(ds, "scar_c")],
                             sarem_drop=base[(ds, "sar_em (oracle)")] - sub[(ds, "sar_em (oracle)")]))
        except KeyError:
            pass
r = pd.DataFrame(rows)
print("\ndrop from the SCAR-mechanism reference (positive = hurt by the bias):")
print(r.pivot_table(index="dataset", columns="mechanism",
                    values=["scar_drop", "sarem_drop"]).round(3).to_string())
print(f"\nmean drop  SCAR-c {r.scar_drop.mean():+.3f}   "
      f"SAR-EM {r.sarem_drop.mean():+.3f}   "
      f"(n = {len(r)} dataset-mechanism cells)")

print("\n" + "=" * 74)
print("D. Ranking against calibration: is the loss real or is it the threshold?")
for metric in ["auc", "auprc", "f1_bestthr", "ece", "brier"]:
    print(f"\n{metric}:")
    print(d.pivot_table(index="label", columns="mechanism", values=metric,
                        aggfunc="mean").reindex(ORDER)[MECHS].round(3).to_string())

print("\n" + "=" * 74)
print("E. SAR-EM (oracle) minus the SCAR baseline, metric by metric")
piv = d.pivot_table(index=["dataset", "mechanism"], columns="label",
                    values=["bacc", "f1", "f1_bestthr", "auc", "auprc", "ece"],
                    aggfunc="mean")
gaps = {}
for metric in ["bacc", "f1", "f1_bestthr", "auc", "auprc", "ece"]:
    gaps[metric] = (piv[(metric, "sar_em (oracle)")]
                    - piv[(metric, "scar_c")]).groupby("mechanism").mean()
print(pd.DataFrame(gaps).reindex(MECHS).round(3).to_string())
print("\nA negative bacc gap next to a near-zero AUC gap means the two models")
print("rank alike and differ only in where 0.5 falls; ECE says which is closer")
print("to being a probability.")

print("\nEM iterations to convergence (median):")
print(d[d.method == "sar_em"].groupby(["prop_spec", "mechanism"])["iters"]
      .median().unstack().round(0).to_string())
