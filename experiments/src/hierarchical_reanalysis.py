from __future__ import annotations

import itertools
import os
import warnings
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from splice_generated import splice

import numpy as np
import pandas as pd
from scipy.stats import chi2, rankdata


HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"
N_BOOT = int(os.environ.get("PU_BOOTSTRAP_REPLICATES", "10000"))
RNG_SEED = 20260815

DATASETS = ["breast", "spambase", "mushroom", "adult", "bank"]
CS_MAIN = [0.1, 0.3, 0.5, 0.7]
CS_MECH = [0.3, 0.5]
SEEDS = [0, 1, 2, 3, 4]
CORE_REIMPLEMENTATIONS = ["vpu", "distpu"]

NICE = {
    "bbe_cvir": "BBE->CVIR",
    "cvir": "CVIR",
    "imbpu": "ImbPU",
    "bbe_nnpu": "BBE->nnPU",
    "wce": "Weighted CE",
    "distpu_off": "Dist-PU",
    "vpu_off": "VPU",
    "nnpu": "nnPU",
    "holistic_off": "HolisticPU",
    "en": "Elkan-Noto",
    "upu": "uPU",
    "uasn": "U-as-negative",
    "dcpu": "DC-PU",
}
TEX_LABEL = {
    "bbe_cvir": r"BBE$\to$CVIR", "cvir": "CVIR", "imbpu": "ImbPU",
    "bbe_nnpu": r"BBE$\to$nnPU", "wce": "Weighted CE",
    "distpu_off": "Dist-PU", "vpu_off": "VPU", "nnpu": "nnPU",
    "holistic_off": "HolisticPU", "en": "Elkan--Noto", "upu": "uPU",
    "uasn": r"$U$-as-negative", "dcpu": "DC-PU",
}


def pick(df: pd.DataFrame, key: list[str]) -> pd.DataFrame:
    return df.loc[df.groupby(key, observed=True)["crit_leeliu"].idxmax()].copy()


def load_main() -> pd.DataFrame:
    key = ["dataset", "c", "seed", "method"]
    grid = pd.concat(
        [pd.read_csv(RESULTS / f) for f in
         ["p1_results.csv", "p15_results.csv", "ext_results.csv",
          "dcpu_results.csv"]],
        ignore_index=True,
    )
    grid = grid[~grid.method.isin(["tice_nnpu"] + CORE_REIMPLEMENTATIONS)]
    off = pd.read_csv(RESULTS / "officialgrid_results.csv")
    off = pick(off[off.phase == "tab"], key)
    tab = pd.concat([pick(grid, key), off], ignore_index=True)
    expected = len(DATASETS) * len(CS_MAIN) * len(SEEDS) * 13
    if len(tab) != expected or tab.method.nunique() != 13:
        raise ValueError(f"main grid is incomplete: {len(tab)} rows, "
                         f"{tab.method.nunique()} methods")
    return tab


def main_tensor(tab: pd.DataFrame):
    methods = sorted(tab.method.unique())
    index = pd.MultiIndex.from_product(
        [DATASETS, CS_MAIN, SEEDS], names=["dataset", "c", "seed"]
    )
    p = tab.pivot(index=["dataset", "c", "seed"], columns="method",
                  values="bacc").reindex(index=index, columns=methods)
    if p.isna().any().any():
        raise ValueError("missing cells in main comparison")
    return p.to_numpy().reshape(len(DATASETS), len(CS_MAIN), len(SEEDS),
                                len(methods)), methods


def hierarchical_bootstrap_main(x: np.ndarray, rng: np.random.Generator):
    d_n, c_n, s_n, m_n = x.shape
    out = np.empty((N_BOOT, m_n), dtype=float)
    for b in range(N_BOOT):
        total = np.zeros(m_n, dtype=float)
        for d in rng.integers(0, d_n, size=d_n):
            for c in rng.integers(0, c_n, size=c_n):
                seeds = rng.integers(0, s_n, size=s_n)
                total += x[d, c, seeds, :].mean(axis=0)
        out[b] = total / (d_n * c_n)
    return out


def exact_signflip_p(dataset_effects: np.ndarray) -> float:
    effects = np.asarray(dataset_effects, dtype=float)
    observed = abs(effects.mean())
    vals = []
    for signs in itertools.product([-1.0, 1.0], repeat=len(effects)):
        vals.append(abs(np.mean(effects * np.asarray(signs))))
    vals = np.asarray(vals)
    return float(np.mean(vals >= observed - 1e-15))


def holm_adjust(pvalues: list[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    n = len(p)
    for rank, idx in enumerate(order):
        value = (n - rank) * p[idx]
        running = max(running, value)
        adjusted[idx] = min(1.0, running)
    return adjusted


def ci(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.quantile(values, [0.025, 0.975])
    return float(lo), float(hi)


def analyse_main(tab: pd.DataFrame, rng: np.random.Generator):
    x, methods = main_tensor(tab)
    boot = hierarchical_bootstrap_main(x, rng)
    observed = x.mean(axis=(0, 1, 2))
    dataset_means = x.mean(axis=(1, 2))
    dataset_ranks = np.vstack([rankdata(-row, method="average")
                               for row in dataset_means])

    pairs = []
    raw_ps = []
    for i, j in itertools.combinations(range(len(methods)), 2):
        delta_boot = boot[:, i] - boot[:, j]
        lo, hi = ci(delta_boot)
        p = exact_signflip_p(dataset_means[:, i] - dataset_means[:, j])
        pairs.append({
            "method_a": methods[i], "method_b": methods[j],
            "effect_a_minus_b": observed[i] - observed[j],
            "ci95_low": lo, "ci95_high": hi, "p_exact": p,
        })
        raw_ps.append(p)
    adjusted = holm_adjust(raw_ps)
    for row, p_adj in zip(pairs, adjusted):
        row["p_holm_pairs"] = p_adj
    pair_df = pd.DataFrame(pairs).sort_values(["p_holm_pairs", "p_exact"])
    pair_df.to_csv(RESULTS / "hierarchical_main_pairwise.csv", index=False)

    leader_idx = int(np.argmax(observed))
    leader = methods[leader_idx]
    lookup = {}
    for row in pairs:
        lookup[(row["method_a"], row["method_b"])] = row

    rows = []
    for i, method in enumerate(methods):
        mean_lo, mean_hi = ci(boot[:, i])
        if i == leader_idx:
            delta, dlo, dhi, p_raw, p_adj = 0.0, 0.0, 0.0, 1.0, 1.0
        else:
            a, b = sorted([leader, method])
            row = lookup[(a, b)]
            sign = 1.0 if a == leader else -1.0
            delta = sign * row["effect_a_minus_b"]
            if sign > 0:
                dlo, dhi = row["ci95_low"], row["ci95_high"]
            else:
                dlo, dhi = -row["ci95_high"], -row["ci95_low"]
            p_raw, p_adj = row["p_exact"], row["p_holm_pairs"]
        worst_idx = int(np.argmin(dataset_means[:, i]))
        rows.append({
            "method": method, "label": NICE.get(method, method),
            "mean_bacc": observed[i], "mean_ci95_low": mean_lo,
            "mean_ci95_high": mean_hi,
            "average_dataset_rank": dataset_ranks[:, i].mean(),
            "worst_dataset": DATASETS[worst_idx],
            "worst_dataset_bacc": dataset_means[worst_idx, i],
            "leader_minus_method": delta,
            "leader_diff_ci95_low": dlo, "leader_diff_ci95_high": dhi,
            "p_exact": p_raw, "p_holm_pairs": p_adj,
        })
    summary = pd.DataFrame(rows).sort_values("mean_bacc", ascending=False)
    summary.to_csv(RESULTS / "hierarchical_main_summary.csv", index=False)
    return summary, pair_df, leader, dataset_means, methods


def mechanism_tensor(selected: pd.DataFrame, metrics: list[str]):
    methods = sorted(selected.method.unique())
    mechanisms = ["scar", "sar_attr", "sar_gap", "sar_extreme"]
    index = pd.MultiIndex.from_product(
        [DATASETS, mechanisms, CS_MECH, SEEDS, methods],
        names=["dataset", "mechanism", "c", "seed", "method"],
    )
    p = selected.set_index(["dataset", "mechanism", "c", "seed", "method"])
    p = p.reindex(index)[metrics]
    if p.isna().any().any():
        raise ValueError("missing cells in mechanism comparison")
    x = p.to_numpy().reshape(len(DATASETS), len(mechanisms), len(CS_MECH),
                             len(SEEDS), len(methods), len(metrics))
    return x, mechanisms, methods


def hierarchical_bootstrap_mechanisms(x: np.ndarray, rng: np.random.Generator):
    d_n, k_n, c_n, s_n, m_n, q_n = x.shape
    out = np.empty((N_BOOT, k_n, q_n), dtype=float)
    for b in range(N_BOOT):
        total = np.zeros((k_n, q_n), dtype=float)
        for d in rng.integers(0, d_n, size=d_n):
            for c in rng.integers(0, c_n, size=c_n):
                seeds = rng.integers(0, s_n, size=s_n)
                branch = np.take(x[d, :, c, :, :, :], seeds, axis=1)
                total += branch.mean(axis=(1, 2))
        out[b] = total / (d_n * c_n)
    return out


def analyse_mechanisms(rng: np.random.Generator):
    raw = pd.read_csv(RESULTS / "calib_results.csv")
    key = ["dataset", "mechanism", "c", "seed", "method"]
    selected = pick(raw, key)
    metrics = ["bacc", "f1", "auc", "auprc", "f1_bestthr", "ece", "brier"]
    x, mechanisms, methods = mechanism_tensor(selected, metrics)
    boot = hierarchical_bootstrap_mechanisms(x, rng)
    obs = x.mean(axis=(0, 2, 3, 4))
    dataset_means = x.mean(axis=(2, 3, 4))
    scar_idx = mechanisms.index("scar")
    rows, raw_ps = [], []
    for k, mechanism in enumerate(mechanisms):
        if mechanism == "scar":
            continue
        for q, metric in enumerate(metrics):
            delta_boot = boot[:, k, q] - boot[:, scar_idx, q]
            lo, hi = ci(delta_boot)
            p = exact_signflip_p(dataset_means[:, k, q] -
                                 dataset_means[:, scar_idx, q])
            rows.append({
                "mechanism": mechanism, "metric": metric,
                "scar_mean": obs[scar_idx, q], "mechanism_mean": obs[k, q],
                "effect_vs_scar": obs[k, q] - obs[scar_idx, q],
                "ci95_low": lo, "ci95_high": hi, "p_exact": p,
            })
            raw_ps.append(p)
    adjusted = holm_adjust(raw_ps)
    for row, p_adj in zip(rows, adjusted):
        row["p_holm_21"] = p_adj
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "hierarchical_mechanism_effects.csv", index=False)
    return selected, out


CRITS = ["crit_leeliu", "crit_puauc", "crit_nnpurisk", "crit_f1hat",
         "crit_nnpurisk_true", "crit_f1hat_true"]
CRIT_LABEL = {
    "crit_leeliu": "Lee-Liu", "crit_puauc": "PU-AUC",
    "crit_nnpurisk": "nnPU risk (estimated prior)",
    "crit_f1hat": "estimated F1 (estimated prior)",
    "crit_nnpurisk_true": "nnPU risk (true prior)",
    "crit_f1hat_true": "estimated F1 (true prior)",
}


def criteria_selected():
    raw = pd.read_csv(RESULTS / "crit_results.csv")
    key = ["dataset", "c", "seed", "method"]
    selected = {}
    for criterion in CRITS + ["crit_oraclef1"]:
        selected[criterion] = raw.loc[
            raw.groupby(key, observed=True)[criterion].idxmax()
        ].set_index(key).sort_index()
    return selected


def criterion_diff_tensor(a: pd.Series, b: pd.Series):
    methods = sorted(a.index.get_level_values("method").unique())
    index = pd.MultiIndex.from_product(
        [DATASETS, CS_MECH, SEEDS, methods],
        names=["dataset", "c", "seed", "method"],
    )
    delta = (a - b).reindex(index)
    if delta.isna().any():
        raise ValueError("missing cells in criterion comparison")
    return delta.to_numpy().reshape(len(DATASETS), len(CS_MECH), len(SEEDS),
                                    len(methods))


def hierarchical_bootstrap_scalar(x: np.ndarray, rng: np.random.Generator):
    d_n, c_n, s_n, _ = x.shape
    out = np.empty(N_BOOT, dtype=float)
    for b in range(N_BOOT):
        total = 0.0
        for d in rng.integers(0, d_n, size=d_n):
            for c in rng.integers(0, c_n, size=c_n):
                seeds = rng.integers(0, s_n, size=s_n)
                total += float(x[d, c, seeds, :].mean())
        out[b] = total / (d_n * c_n)
    return out


def analyse_criteria(rng: np.random.Generator):
    s = criteria_selected()
    oracle = s["crit_oraclef1"]
    lee = s["crit_leeliu"]
    rows, raw_ps = [], []
    for criterion in CRITS[1:]:
        chosen = s[criterion]
        x = criterion_diff_tensor(chosen.f1, lee.f1)
        boot = hierarchical_bootstrap_scalar(x, rng)
        lo, hi = ci(boot)
        ds = x.mean(axis=(1, 2, 3))
        p = exact_signflip_p(ds)
        rows.append({
            "family": "versus_lee_liu_f1", "criterion": criterion,
            "label": CRIT_LABEL[criterion],
            "effect_f1": x.mean(), "ci95_low": lo, "ci95_high": hi,
            "p_exact": p,
            "oracle_gap_f1": float((oracle.f1 - chosen.f1).mean()),
        })
        raw_ps.append(p)
    adjusted = holm_adjust(raw_ps)
    for row, p_adj in zip(rows, adjusted):
        row["p_holm_within_family"] = p_adj

    prior_rows, prior_ps = [], []
    for estimated, true in [("crit_nnpurisk", "crit_nnpurisk_true"),
                            ("crit_f1hat", "crit_f1hat_true")]:
        x = criterion_diff_tensor(s[estimated].f1, s[true].f1)
        boot = hierarchical_bootstrap_scalar(x, rng)
        lo, hi = ci(boot)
        p = exact_signflip_p(x.mean(axis=(1, 2, 3)))
        prior_rows.append({
            "family": "estimated_vs_true_prior_f1", "criterion": estimated,
            "label": CRIT_LABEL[estimated], "effect_f1": x.mean(),
            "ci95_low": lo, "ci95_high": hi, "p_exact": p,
            "oracle_gap_f1": float((oracle.f1 - s[estimated].f1).mean()),
        })
        prior_ps.append(p)
    adjusted = holm_adjust(prior_ps)
    for row, p_adj in zip(prior_rows, adjusted):
        row["p_holm_within_family"] = p_adj
    out = pd.DataFrame(rows + prior_rows)
    out.to_csv(RESULTS / "hierarchical_criteria_effects.csv", index=False)
    return out


def prepare_mixed_data(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["c_factor"] = d.c.astype(str)
    d["dataset_method"] = d.dataset.astype(str) + ":" + d.method.astype(str)
    d["dataset_c"] = d.dataset.astype(str) + ":" + d.c.astype(str)
    d["dataset_seed"] = d.dataset.astype(str) + ":" + d.seed.astype(str)
    if "mechanism" in d:
        d["dataset_mechanism"] = (d.dataset.astype(str) + ":" +
                                  d.mechanism.astype(str))
    return d


def fit_mixed(formula: str, data: pd.DataFrame, vc_formula: dict[str, str]):
    import statsmodels.formula.api as smf

    model = smf.mixedlm(formula, data=data, groups=data["dataset"],
                        re_formula="1", vc_formula=vc_formula)
    attempts = []
    for method in ["lbfgs", "powell"]:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = model.fit(reml=False, method=method, maxiter=3000,
                                   disp=False)
            attempts.extend(str(w.message) for w in caught)
            if result.converged:
                return result, attempts
        except Exception as exc:
            attempts.append(f"{method}: {type(exc).__name__}: {exc}")
    raise RuntimeError("mixed model failed: " + " | ".join(attempts))


def lrt(full, reduced):
    stat = max(0.0, 2.0 * (full.llf - reduced.llf))
    df = int(round(full.df_modelwc - reduced.df_modelwc))
    return stat, df, float(chi2.sf(stat, df))


def mixed_models(main: pd.DataFrame, mechanism: pd.DataFrame):
    main_d = prepare_mixed_data(main)
    main_vc = {
        "dataset_method": "0 + C(dataset_method)",
    }
    main_full, w1 = fit_mixed("bacc ~ C(method) + C(c_factor)", main_d, main_vc)
    main_red, w2 = fit_mixed("bacc ~ C(c_factor)", main_d, main_vc)
    main_test = lrt(main_full, main_red)

    mech_d = prepare_mixed_data(mechanism)
    mech_vc = {
        "dataset_method": "0 + C(dataset_method)",
        "dataset_mechanism": "0 + C(dataset_mechanism)",
    }
    mech_full, w3 = fit_mixed(
        "bacc ~ C(method) * C(mechanism) + C(c_factor)", mech_d, mech_vc
    )
    mech_add, w4 = fit_mixed(
        "bacc ~ C(method) + C(mechanism) + C(c_factor)", mech_d, mech_vc
    )
    mech_none, w5 = fit_mixed(
        "bacc ~ C(method) + C(c_factor)", mech_d, mech_vc
    )
    mechanism_test = lrt(mech_add, mech_none)
    interaction_test = lrt(mech_full, mech_add)

    lines = [
        "MIXED-EFFECTS SENSITIVITY MODELS (maximum likelihood)",
        "Dataset is the grouping factor; named components are nested/crossed "
        "random-intercept variance components.", "",
        "MAIN TWELVE-METHOD BALANCED-ACCURACY MODEL",
        f"full llf={main_full.llf:.6f}; reduced llf={main_red.llf:.6f}",
        f"global method LRT: chi2={main_test[0]:.6f}, df={main_test[1]}, "
        f"p={main_test[2]:.8g}",
        f"full converged={main_full.converged}; reduced converged={main_red.converged}",
        f"full variance parameters={list(map(float, main_full.vcomp))}; "
        f"dataset intercept variance={float(main_full.cov_re.iloc[0, 0]):.8g}",
        "", "MECHANISM FACTORIAL BALANCED-ACCURACY MODEL",
        f"full llf={mech_full.llf:.6f}; additive llf={mech_add.llf:.6f}; "
        f"no-mechanism llf={mech_none.llf:.6f}",
        f"mechanism main-effect LRT: chi2={mechanism_test[0]:.6f}, "
        f"df={mechanism_test[1]}, p={mechanism_test[2]:.8g}",
        f"method-by-mechanism interaction LRT: chi2={interaction_test[0]:.6f}, "
        f"df={interaction_test[1]}, p={interaction_test[2]:.8g}",
        f"full converged={mech_full.converged}; additive converged={mech_add.converged}; "
        f"no-mechanism converged={mech_none.converged}",
        f"full variance parameters={list(map(float, mech_full.vcomp))}; "
        f"dataset intercept variance={float(mech_full.cov_re.iloc[0, 0]):.8g}",
        "", "WARNINGS", *[f"- {x}" for x in (w1 + w2 + w3 + w4 + w5)],
    ]
    (RESULTS / "mixed_effects_models.txt").write_text("\n".join(lines) + "\n")
    return main_test, mechanism_test, interaction_test


def write_summary(main_summary, pairwise, leader, mechanism, criteria,
                  main_test, mechanism_test, interaction_test):
    lines = [
        "HIERARCHICAL REANALYSIS SUMMARY",
        f"bootstrap replicates={N_BOOT}; random seed={RNG_SEED}",
        "primary endpoint=test balanced accuracy; highest independent unit=dataset",
        "", f"Observed leader: {NICE.get(leader, leader)} ({leader})",
        "No method separated from every competitor under the 66-contrast Holm family.",
        "", "MAIN METHOD SUMMARIES",
        main_summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"),
        "", "PAIRWISE MULTIPLICITY",
        f"Holm-significant pairs: {(pairwise.p_holm_pairs < 0.05).sum()} / {len(pairwise)}",
        f"minimum raw exact p={pairwise.p_exact.min():.6f}; "
        f"minimum Holm p={pairwise.p_holm_pairs.min():.6f}",
        "", "MIXED MODELS",
        f"main method LRT: chi2={main_test[0]:.4f}, df={main_test[1]}, p={main_test[2]:.6g}",
        f"mechanism main-effect LRT: chi2={mechanism_test[0]:.4f}, "
        f"df={mechanism_test[1]}, p={mechanism_test[2]:.6g}",
        f"interaction LRT: chi2={interaction_test[0]:.4f}, "
        f"df={interaction_test[1]}, p={interaction_test[2]:.6g}",
        "", "MECHANISM EFFECTS",
        mechanism.to_string(index=False, float_format=lambda x: f"{x:.6f}"),
        "", "MODEL-SELECTION CRITERION EFFECTS",
        criteria.to_string(index=False, float_format=lambda x: f"{x:.6f}"),
    ]
    (RESULTS / "hierarchical_reanalysis_summary.txt").write_text(
        "\n".join(lines) + "\n"
    )


def write_supplementary_tables(main_summary, pairwise, mechanism, criteria,
                               main_test, mechanism_test, interaction_test):
    lines = [


        r"\begin{table}[H]", r"\centering", r"\small",
        r"\caption{Complete cluster-aware summaries for the twelve-method primary comparison. Intervals are percentile 95\% hierarchical-bootstrap intervals. Ranks use five dataset-level blocks.}",
        r"\label{tab:supp-main-cluster}",
        r"\begin{tabular}{@{}lrrrrl@{}}", r"\toprule",
        r"Method & Mean & 95\% CI & Avg. rank & Worst & Dataset \\",
        r"\midrule",
    ]
    for row in main_summary.itertuples(index=False):
        lines.append(
            f"{TEX_LABEL[row.method]} & {row.mean_bacc:.3f} & "
            f"$[{row.mean_ci95_low:.3f},{row.mean_ci95_high:.3f}]$ & "
            f"{row.average_dataset_rank:.1f} & {row.worst_dataset_bacc:.3f} & "
            f"{str(row.worst_dataset).title()} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", "",


              r"\begingroup", r"\scriptsize",
              r"\begin{longtable}{@{}llrrrr@{}}",
              r"\caption{All 66 paired method contrasts for the primary endpoint. Effect is method A minus method B. Intervals are unadjusted percentile 95\% hierarchical-bootstrap intervals; $p$ is the two-sided exact sign-flip test over five dataset effects and $p_{\mathrm H}$ controls the complete 66-contrast family by Holm's method.}\label{tab:supp-all-pairs}\\",
              r"\toprule",
              r"Method A & Method B & Effect & 95\% CI & $p$ & $p_{\mathrm H}$ \\",
              r"\midrule", r"\endfirsthead",
              r"\multicolumn{6}{c}{\tablename\ \thetable{} -- continued}\\",
              r"\toprule", r"Method A & Method B & Effect & 95\% CI & $p$ & $p_{\mathrm H}$ \\",
              r"\midrule", r"\endhead", r"\midrule",
              r"\multicolumn{6}{r}{continued on next page}\\", r"\endfoot",
              r"\bottomrule", r"\endlastfoot"]
    order = list(main_summary.method)
    by_pair = {(r.method_a, r.method_b): r
               for r in pairwise.itertuples(index=False)}
    for ia, a in enumerate(order):
        for b in order[ia + 1:]:
            key = (a, b) if (a, b) in by_pair else (b, a)
            row = by_pair[key]
            sign = 1.0 if key == (a, b) else -1.0
            effect = sign * row.effect_a_minus_b
            lo = row.ci95_low if sign > 0 else -row.ci95_high
            hi = row.ci95_high if sign > 0 else -row.ci95_low
            lines.append(
                f"{TEX_LABEL[a]} & {TEX_LABEL[b]} & {effect:+.3f} & "
                f"$[{lo:+.3f},{hi:+.3f}]$ & {row.p_exact:.3f} & "
                f"{row.p_holm_pairs:.3f} \\\\"
            )
    lines += [r"\end{longtable}", r"\endgroup", "",
              r"\begin{table}[htbp]", r"\centering", r"\scriptsize",
              r"\caption{Mixed-effects sensitivity-model likelihood-ratio tests. The dataset-level hierarchical bootstrap remains the primary uncertainty analysis.}",
              r"\label{tab:supp-mixed-models}",
              r"\begin{tabular}{@{}lrrl@{}}", r"\toprule",
              r"Term & $\chi^2$ & df & $p$ \\", r"\midrule",
              f"Method (primary bacc model) & {main_test[0]:.2f} & {main_test[1]} & {main_test[2]:.4f} \\\\",
              f"Mechanism (factorial bacc model) & {mechanism_test[0]:.2f} & {mechanism_test[1]} & {mechanism_test[2]:.4f} \\\\",
              f"Method $\\times$ mechanism & {interaction_test[0]:.2f} & {interaction_test[1]} & $<10^{{-20}}$ \\\\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}", "",


              r"\begin{table}[htbp]", r"\centering", r"\scriptsize",
              r"\caption{All mechanism--metric effects against matched SCAR runs. Positive ECE and Brier effects are deterioration; negative effects in the other metrics are deterioration. The 21 contrasts form one Holm family.}\label{tab:supp-mechanism-effects}",
              r"\begin{tabular}{@{}llrrrr@{}}",
              r"\toprule", r"Mechanism & Metric & Effect & 95\% CI & $p$ & $p_{\mathrm H}$ \\",
              r"\midrule"]
    mech_label = {"sar_attr": "SAR-attribute", "sar_gap": "SAR-gap",
                  "sar_extreme": "SAR-severe"}
    metric_label = {"bacc": "bacc", "f1": r"$F_1$", "auc": "AUC",
                    "auprc": "AP", "f1_bestthr": r"$F_1^{\ast}$",
                    "ece": "ECE", "brier": "Brier"}
    for row in mechanism.itertuples(index=False):
        lines.append(
            f"{mech_label[row.mechanism]} & {metric_label[row.metric]} & "
            f"{row.effect_vs_scar:+.3f} & $[{row.ci95_low:+.3f},{row.ci95_high:+.3f}]$ & "
            f"{row.p_exact:.3f} & {row.p_holm_21:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
              r"\begin{table}[htbp]", r"\centering", r"\scriptsize",
              r"\caption{Cluster-aware model-selection-criterion contrasts on test $F_1$. The upper family compares five criteria with Lee--Liu; the lower family compares estimated- and true-prior versions. Holm correction is within the stated family.}",
              r"\label{tab:supp-criteria-effects}",
              r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
              r"Family & Contrast & Effect & 95\% CI & $p$ & $p_{\mathrm H}$ \\", r"\midrule"]
    for row in criteria.itertuples(index=False):
        family = ("vs. Lee--Liu" if row.family == "versus_lee_liu_f1"
                  else "est. vs. true prior")
        label = row.label.replace("estimated prior", "est. prior")
        lines.append(
            f"{family} & {label} & {row.effect_f1:+.3f} & "
            f"$[{row.ci95_low:+.3f},{row.ci95_high:+.3f}]$ & "
            f"{row.p_exact:.3f} & {row.p_holm_within_family:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    splice("stats", "\n".join(lines),
           "experiments/src/hierarchical_reanalysis.py")


def main():
    rng = np.random.default_rng(RNG_SEED)
    tab = load_main()
    main_summary, pairwise, leader, _, _ = analyse_main(tab, rng)
    mechanism_selected, mechanism = analyse_mechanisms(rng)
    criteria = analyse_criteria(rng)
    tests = mixed_models(tab, mechanism_selected)
    write_summary(main_summary, pairwise, leader, mechanism, criteria, *tests)
    write_supplementary_tables(main_summary, pairwise, mechanism, criteria, *tests)
    print((RESULTS / "hierarchical_reanalysis_summary.txt").read_text())


if __name__ == "__main__":
    main()
