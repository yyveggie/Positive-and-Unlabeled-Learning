from __future__ import annotations

import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(HERE, "results")


def pick(df, crit):
    key = ["dataset", "mechanism", "c", "seed", "method"]
    return df.loc[df.groupby(key)[crit].idxmax()]


p15_path = os.path.join(R, "p15_results.csv")
p1_path = os.path.join(R, "p1_results.csv")
if os.path.exists(p15_path) and os.path.exists(p1_path):
    p15 = pd.read_csv(p15_path)
    p1 = pd.read_csv(p1_path)
    print("=" * 70)
    print("P1.5  BBE->nnPU (PU-only selection), vs P1 columns")
    sel = pick(p15, "crit_leeliu")
    row = sel.groupby("dataset")["bacc"].mean()
    ref = pick(p1, "crit_leeliu")
    comp = ref[ref.method.isin(["nnpu", "tice_nnpu", "wce"])] \
        .pivot_table(index="method", columns="dataset", values="bacc", aggfunc="mean")
    comp.loc["bbe_nnpu"] = row
    comp["MEAN"] = comp.mean(axis=1)
    print(comp.round(3).sort_values("MEAN", ascending=False).to_string())
    err = p15.drop_duplicates(subset=["dataset", "c", "seed"])
    err = err.assign(rel=(err.alpha_used - err.alpha_true).abs() / err.alpha_true)
    print("\nBBE prior relative error by dataset & c:")
    print(err.pivot_table(index="dataset", columns="c", values="rel",
                          aggfunc="mean").round(2).to_string())


e3_path = os.path.join(R, "e3_results.csv")
if os.path.exists(e3_path):
    e3 = pd.read_csv(e3_path)
    e3["base"] = e3.method.str.split("_x").str[0]
    e3["factor"] = e3.method.str.split("_x").str[1].astype(float)
    sel = e3.loc[e3.groupby(["dataset", "seed", "method"])["crit_leeliu"].idxmax()]
    print("\n" + "=" * 70)
    print("E3  test bacc vs prior misspecification factor (c=0.5, mean over datasets/seeds)")
    print(sel.pivot_table(index="base", columns="factor", values="bacc",
                          aggfunc="mean").round(3).to_string())
    print("\nper-dataset, nnPU only:")
    print(sel[sel.base == "nnpu"].pivot_table(index="dataset", columns="factor",
          values="bacc", aggfunc="mean").round(3).to_string())
    paired = sel.pivot(
        index=["dataset", "seed", "base"], columns="factor", values="bacc"
    )
    asymmetry = []
    for lower, upper in [(0.75, 1.25), (0.5, 1.5)]:
        difference = paired[upper] - paired[lower]
        asymmetry.append({
            "underestimate_factor": lower,
            "overestimate_factor": upper,
            "overestimate_at_least_as_accurate": int((difference >= 0).sum()),
            "paired_cells": int(len(difference)),
            "proportion": float((difference >= 0).mean()),
            "mean_bacc_difference": float(difference.mean()),
            "independent_dataset_units": int(
                paired.reset_index()["dataset"].nunique()
            ),
        })
    pd.DataFrame(asymmetry).to_csv(
        os.path.join(R, "prior_misspecification_pairs.csv"), index=False
    )


e4_path = os.path.join(R, "e4_results.csv")
if os.path.exists(e4_path) and os.path.exists(p1_path):
    e4 = pd.read_csv(e4_path)
    p1 = pd.read_csv(p1_path)
    sel4 = pick(e4, "crit_leeliu")
    scar = pick(p1[p1.c.isin([0.3, 0.5])], "crit_leeliu")
    scar = scar[scar.method.isin(sel4.method.unique())]
    print("\n" + "=" * 70)
    print("E4  test bacc under SAR mechanisms vs SCAR reference (c in {0.3,0.5})")
    tab = pd.concat([
        scar.assign(mechanism="scar(ref)"),
        sel4])[["mechanism", "method", "bacc"]] \
        .pivot_table(index="method", columns="mechanism", values="bacc", aggfunc="mean")
    for m in ["sar_attr", "sar_gap"]:
        if m in tab:
            tab[f"drop_{m}"] = tab["scar(ref)"] - tab[m]
    print(tab.round(3).to_string())


p2_path = os.path.join(R, "p2_results.csv")
if os.path.exists(p2_path):
    p2 = pd.read_csv(p2_path)
    key = ["dataset", "c", "seed", "method"]
    pu_sel = p2.loc[p2.groupby(key)["crit_leeliu"].idxmax()]
    or_sel = p2.loc[p2.groupby(key)["crit_oraclef1"].idxmax()]
    m = or_sel.merge(pu_sel, on=key, suffixes=("_or", "_pu"))
    m["gap_f1"] = m.f1_or - m.f1_pu
    m["gap_bacc"] = m.bacc_or - m.bacc_pu
    print("\n" + "=" * 70)
    print(f"P2  oracle gap with expanded search space "
          f"({p2.groupby(key).size().max()} configs/method vs 4 in P1)")
    print(m.groupby("method")[["gap_f1", "gap_bacc"]]
          .agg(["mean", "std"]).round(4).to_string())
    print("\nepoch chosen by each criterion (mean):")
    print(pd.DataFrame({"pu_only": pu_sel.groupby("method")["epoch"].mean(),
                        "oracle": or_sel.groupby("method")["epoch"].mean()}).round(1).to_string())
