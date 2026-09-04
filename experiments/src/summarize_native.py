from __future__ import annotations

import os
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(HERE, "results")
d = pd.read_csv(os.path.join(R, "native_results.csv"))
main = d[d.phase == "main"]
sel = main.loc[main.groupby(["seed", "method"])["crit_leeliu"].idxmax()]
PRIOR_FREE = {"uasn", "wce", "vpu_off", "holistic_off"}
COLS = ["bacc", "f1", "auc", "auprc", "f1_bestthr", "ece", "brier"]

a = d.alpha_true.mean()
c = d.c_true.mean()
print(f"Swiss-Prot human nuclear localization: alpha = {a:.4f}, c = {c:.4f}, "
      f"{len(d)} runs, {sel.seed.nunique()} seeds\n")

print("=" * 78)
print("A. Test metrics after PU-only model selection (mean over seeds)")
t = sel.groupby("method")[COLS].mean()
t.insert(0, "prior-free", [m in PRIOR_FREE for m in t.index])
t = t.sort_values("bacc", ascending=False)
print(t.round(3).to_string())
print(f"\n  {len(t)} methods in the roster")

print("\n" + "=" * 78)
print("B. Class-prior estimation where nobody chose the prior")
b = sel[sel.method == "bbe_nnpu"]
if len(b):
    print(f"  true alpha      {b.alpha_true.mean():.4f}")
    print(f"  BBE estimate    {b.alpha_used.mean():.4f}   "
          f"ratio {b.alpha_used.mean() / b.alpha_true.mean():.2f}")
    print(f"  relative error  "
          f"{abs(b.alpha_used.mean() - b.alpha_true.mean()) / b.alpha_true.mean():.2f}")

pr = d[d.phase == "prior"]
if len(pr):
    pr = pr.copy()
    pr["k"] = pr.method.str.split("_x").str[1].astype(float)
    ps = pr.loc[pr.groupby(["seed", "method"])["crit_leeliu"].idxmax()]
    print("\n" + "=" * 78)
    print("C. Does 'err upward' hold on a dataset we did not construct?")
    g = ps.groupby("k")[["bacc", "f1", "auc", "ece"]].mean()
    print(g.round(3).to_string())
    best = g.bacc.idxmax()
    print(f"\n  balanced accuracy peaks at k = {best} "
          f"({g.bacc.max():.3f} against {g.bacc.loc[1.0]:.3f} at the true prior)")

st = os.path.join(R, "native_results_scartest.csv")
if os.path.exists(st):
    s = pd.read_csv(st)
    print("\n" + "=" * 78)
    print("D. Is the labeling SCAR? The test's verdict, and the truth")
    t1 = s[s.stat != "oracle_permutation"]
    print(t1.groupby("stat")[["p", "reject", "purity"]].mean().round(3).to_string())
    o = s[s.stat == "oracle_permutation"]
    if len(o):
        r = o.iloc[0]
        print(f"\n  ground-truth permutation test on the real partition "
              f"({int(r.n_labeled)} labeled positives against "
              f"{int(r.n_hidden)} hidden ones):")
        print(f"    KS = {r.T0:.3f}, null mean {r.null_mean:.3f}, p = {r.p:.3f}")
        print(f"    purity of the reconstructed positive set: "
              f"{t1.purity.mean():.3f}")
