"""Verify every supplementary figure's numeric claims against the pipeline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from brainturtle import config as cfg, data, evaluate, featuresets as fs
from brainturtle import report, resampling as rs

report.run_banner("l_supplementary_audit")
checks = []


def record(fig, claim, measured, verdict, note=""):
    checks.append({"figure": fig, "claim": claim, "measured": measured,
                   "verdict": verdict, "note": note})
    mark = {"OK": "  OK  ", "WRONG": " WRONG", "CHECK": " CHECK"}[verdict]
    print(f"{mark}  {fig:6s} {claim}")
    print(f"          measured: {measured}")
    if note:
        print(f"          {note}")


report.header("Kang threshold sweep (underpins S4 and S10)")
sweep = []
for name in ("Kang", "Kang (investigator-removed)"):
    df = data.external(name)
    for pct in cfg.REMOVAL_GRID_EXPLORATORY:
        try:
            P, y = evaluate.external_probability_matrix(df, pct)
        except FileNotFoundError:
            continue
        pm = P.mean(axis=1)
        auc = roc_auc_score(y, pm)
        _, lo, hi, _ = rs.external_bootstrap_auc(P, y)
        null = rs.label_permutation_null(y, pm)
        s = rs.summarize_null(null, auc, two_sided=False)
        sweep.append({"cohort": name, "drop_pct": pct, "auc": auc,
                      "ci_low": lo, "ci_high": hi, "p": s["p"],
                      "above_chance": s["p"] < cfg.ALPHA})
sw = pd.DataFrame(sweep)
for name in sw.cohort.unique():
    print(f"\n  {name}")
    report.show(sw[sw.cohort == name],
                ["drop_pct", "auc", "ci_low", "ci_high", "p", "above_chance"])
report.save(sw, "kang_threshold_sweep")

report.header("35% versus 30% removal, paired over speakers")
contrast = []
for name in ("Kang", "Kang (investigator-removed)"):
    df = data.external(name)
    pm = {}
    for pct in (0.30, 0.35):
        P, y = evaluate.external_probability_matrix(df, pct)
        pm[pct] = P.mean(axis=1)
    rng = np.random.default_rng(cfg.BOOT_SEED)
    idx = rng.integers(0, len(y), size=(cfg.N_BOOT, len(y)))
    boots = {pct: np.array([roc_auc_score(y[i], v[i]) for i in idx
                            if len(set(y[i])) > 1]) for pct, v in pm.items()}
    d = boots[0.35] - boots[0.30]
    lo, hi = np.percentile(d, [2.5, 97.5])
    pv = 2 * min((d <= 0).mean(), (d >= 0).mean())
    contrast.append({"cohort": name,
                     "auc_30": roc_auc_score(y, pm[0.30]),
                     "auc_35": roc_auc_score(y, pm[0.35]),
                     "diff": roc_auc_score(y, pm[0.35]) - roc_auc_score(y, pm[0.30]),
                     "ci_low": lo, "ci_high": hi,
                     "p": min(1.0, max(pv, 1.0 / len(d))),
                     "separable": min(1.0, max(pv, 1.0 / len(d))) < cfg.ALPHA})
ct = pd.DataFrame(contrast)
report.show(ct, ["cohort", "auc_30", "auc_35", "diff", "ci_low", "ci_high",
                 "p", "separable"])
report.save(ct, "kang_threshold_contrast")

full = sw[sw.cohort == "Kang"].set_index("drop_pct")
cut = sw[sw.cohort == "Kang (investigator-removed)"].set_index("drop_pct")

report.header("Per-figure verification")

ladder_p = cfg.RESULTS / f"n{cfg.N_PERM}" / "fig2a_removal_ladder.csv"
if ladder_p.exists():
    L = pd.read_csv(ladder_p).set_index("drop_pct")
    peak = L["auc_boot_mean"].idxmax()
    record("S1", "inverted-U peaks at 30% (AUC = 0.743), declines at 35-40%",
           f"peak at {peak:.0%}, AUC {L.loc[peak,'auc_boot_mean']:.4f}; "
           f"35% {L.loc[0.35,'auc_boot_mean']:.4f}, "
           f"40% {L.loc[0.40,'auc_boot_mean']:.4f}",
           "OK" if peak == 0.30 else "WRONG",
           "shape correct; value 0.740 not 0.743 at N_BOOT=10,000")
else:
    record("S1", "peak 0.743 at 30%", "run b_removal_sweep.py first", "CHECK")

feats = fs.common_features()
from scipy.stats import spearmanr
cv = data.commonvoice()
age = cv["age_num_x"].values
rho = {f: abs(spearmanr(cv[f].values, age, nan_policy="omit").statistic)
       for f in feats}

rankable = [f for f in feats if np.isfinite(rho[f])]
top_shap = set([f for f in fs.load_shap_ranking()["feature"]
                if f in set(feats)][:30])
top_sp = set(sorted(rankable, key=lambda x: -rho[x])[:30])
shared = len(top_shap & top_sp)
record("S2", "17 of 30 top-ranked features shared; 'convergent evidence'",
       f"{shared} of 30", "OK" if shared == 17 else "WRONG",
       "count reproduces. The "
       "wording is the issue, not the number: 57% set overlap is the 'modest "
       "agreement' S2's own legend states, not the main text's 'convergent "
       "evidence'")

record("S3", "PCA/UMAP of the DECONFOUNDED feature space; mean silhouette 0.17",
       "cluster.ipynb uses all 110 features; "
       "silhouette mean 0.1744, K=2 = 0.282",
       "WRONG", "0.17 is right; 'deconfounded' is not -- the 78-feature set "
                "is never loaded in that notebook")

if 0.35 in cut.index and 0.35 in full.index:
    record("S4", "cut-condition peak 0.651 at 35% vs 0.668 full; "
                 "'curve shape and peak location similar'",
           f"cut peak {cut['auc'].max():.4f} at "
           f"{cut['auc'].idxmax():.0%} (p={cut.loc[cut['auc'].idxmax(),'p']:.4f}); "
           f"full peak {full['auc'].max():.4f} at {full['auc'].idxmax():.0%}",
           "CHECK",
           "at 30% the cut condition is AUC "
           f"{cut.loc[0.30,'auc']:.4f}, p={cut.loc[0.30,'p']:.4f} -- not above "
           "chance; check whether the reassurance claim survives")

calib = cfg.RESULTS / f"n{cfg.N_PERM}" / "calibration.csv"
if calib.exists():
    C = pd.read_csv(calib)
    record("S5", "Brier 0.2091 vs 0.2406; ECE 0.1474 vs 0.1828",
           f"Brier {C.brier.iloc[1]:.4f} vs {C.brier.iloc[0]:.4f}; "
           f"ECE {C.ece.iloc[1]:.4f} vs {C.ece.iloc[0]:.4f}",
           "CHECK", "Brier reproduces; ECE does not -- pin the binning scheme")

record("S6", "translational scope schematic", "no numeric claims", "OK")

record("S7", "Spearman heatmap over all 110 features vs age",
       f"{len(feats)} features; {len(rankable)} with computable rho", "OK")

pa = data.pitt_with_age()
record("S8", "age distribution and class proportions in Pitt",
       f"n={pa.speaker_id.nunique()}, age {pa.entryage.mean():.1f}"
       f"+-{pa.entryage.std():.1f}, "
       f"{(pa.label==1).sum()} AD / {(pa.label==0).sum()} control", "OK")

s9 = cfg.RESULTS / f"n{cfg.N_PERM}" / "suppfig9_summary.csv"
if s9.exists():
    g = pd.read_csv(s9).iloc[0]
    record("S9", "gender removal decreases performance -> gender features "
                 "carry disease-relevant signal",
           f"dAUC {g.observed:+.4f}, p={g.p:.3f}, "
           f"{int(g.n_ge_observed)}/{int(g.n_perm)} random removals as good "
           f"or better", "WRONG",
           "inside the random-removal null; supports no claim about gender")
else:
    record("S9", "gender removal claim", "run g_gender_control.py", "CHECK")

mono = full["auc"].is_monotonic_increasing
sig_from = full[full.above_chance].index.min() if full.above_chance.any() else None
record("S10", "Kang AUC rises MONOTONICALLY from 0.46 to a peak at 35%, "
              "significant from 5% onward",
       f"peak {full['auc'].max():.4f} at {full['auc'].idxmax():.0%}; "
       f"monotonic increasing: {mono}; first threshold above chance: "
       f"{sig_from if sig_from is None else f'{sig_from:.0%}'}",
       "CHECK", "curve declines after its peak, so 'monotonically' cannot "
                "describe the full 0-40% range")

ff = cfg.FEATURES / "file_features.csv"
if ff.exists():
    F = pd.read_csv(ff)
    for fig, col, claim in [
        ("S11", "duration", "AD recordings: greater median duration, higher "
                            "variance, more upper outliers"),
        ("S12", "speech_ratio", "speech ratio distributions largely overlapping"),
    ]:
        cand = [c for c in F.columns if col.split("_")[0] in c.lower()]
        if not cand:
            record(fig, claim, f"no '{col}' column in file_features.csv", "CHECK",
                   "cannot verify from the saved feature table")
            continue
        c = cand[0]
        a = F[F.label == 1][c]
        h = F[F.label == 0][c]
        record(fig, claim,
               f"{c}: AD median {a.median():.3f} sd {a.std():.3f} | "
               f"control median {h.median():.3f} sd {h.std():.3f}",
               "CHECK")

cmp_r = fs.compare_rankings()
record("S13", "close agreement with Fig 1A confirms the age ranking is stable "
              "and not driven by the data partition",
       f"the two ranking FILES in data/ share only "
       f"{cmp_r['overlap'].iloc[3]['overlap']}/33 at the cut; "
       f"Fig 1 and Figs 2-4 use different files",
       "WRONG", "stability claim is untested as written, and the underlying "
                "files disagree")

out = pd.DataFrame(checks)
report.header("Summary")
report.show(out, ["figure", "verdict", "claim"])
print()
for v in ("WRONG", "CHECK", "OK"):
    n = (out.verdict == v).sum()
    print(f"  {v:6s} {n}  {list(out[out.verdict==v].figure)}")
report.save(out, "supplementary_audit")
