"""Figure 4 — external validation on ADReSSo 2021 and the Korean Kang Corpus."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import roc_auc_score

from brainturtle import config as cfg, data, evaluate, report, resampling as rs

report.run_banner("f_external")

rows = []
for name, spec in data.external_cohorts():
    df = data.external(name)
    P, y = evaluate.external_probability_matrix(df, cfg.PRIMARY_THRESHOLD)
    p_mean = P.mean(axis=1)
    auc = roc_auc_score(y, p_mean)

    y_dup = np.concatenate([y] * P.shape[1])
    p_dup = np.concatenate([P[:, j] for j in range(P.shape[1])])
    auc_dup = roc_auc_score(y_dup, p_dup)
    _, lo_dup, hi_dup, _ = rs.bootstrap_auc(y_dup, p_dup)

    _, lo, hi, _ = rs.external_bootstrap_auc(P, y)

    def make_null(seed, n_perm, _y=y, _p=p_mean):
        return rs.label_permutation_null(_y, _p, n_perm=n_perm, seed=seed)

    null = make_null(cfg.PERM_SEED, cfg.N_PERM)
    summ = rs.summarize_null(null, auc, two_sided=False)
    stab = rs.stability_sweep(make_null, auc, n_perm=cfg.N_PERM_STABILITY,
                              two_sided=False)

    rows.append({
        "cohort": name, "role": spec["role"], "n_speakers": len(y),
        "n_ad": int(y.sum()), "n_control": int((1 - y).sum()),
        "auc": auc, "ci_low": lo, "ci_high": hi, "ci_width": hi - lo,
        "auc_notebook_pooled": auc_dup,
        "ci_low_notebook": lo_dup, "ci_high_notebook": hi_dup,
        "ci_width_notebook": hi_dup - lo_dup,
        "ci_width_ratio": (hi - lo) / (hi_dup - lo_dup),
        "null_mean": summ["null_mean"], "null_sd": summ["null_sd"],
        "n_ge_observed": summ["n_ge_observed"], "p": summ["p"],
        "p_formatted": summ["p_formatted"],
        "p_min_stability": min(r["p"] for r in stab),
        "p_max_stability": max(r["p"] for r in stab),
    })
    print(f"  {name:30s} n={len(y):3d}  AUC={auc:.4f}  {summ['p_formatted']}")

import pandas as pd
res = pd.DataFrame(rows)

report.header("Figure 4 - external validation",
              f"N_PERM={cfg.N_PERM}, N_BOOT={cfg.N_BOOT}, "
              f"threshold={cfg.PRIMARY_THRESHOLD:.0%}")
report.show(res, ["cohort", "role", "n_speakers", "n_ad", "n_control",
                  "auc", "auc_notebook_pooled"])

print()
print("  Bootstrap 95% CI - notebook (seeds as rows) vs speaker-level:")
for r in res.itertuples():
    print(f"    {r.cohort:30s} notebook [{r.ci_low_notebook:.4f}, "
          f"{r.ci_high_notebook:.4f}] w={r.ci_width_notebook:.4f}   "
          f"correct [{r.ci_low:.4f}, {r.ci_high:.4f}] w={r.ci_width:.4f}   "
          f"{r.ci_width_ratio:.2f}x wider")

print()
print("  AUC vs chance (label permutation):")
for r in res.itertuples():
    verdict = "above chance" if r.p < cfg.ALPHA else "NOT above chance"
    print(f"    {r.cohort:30s} {r.auc:.4f}  null {r.null_mean:.4f}  "
          f"{r.n_ge_observed}/{cfg.N_PERM} >= obs  {r.p_formatted}  <- {verdict}")

report.save(res, "fig4_external")
