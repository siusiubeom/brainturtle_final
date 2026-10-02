"""Figure 2a and Supplementary S1 — the removal ladder."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.metrics import roc_auc_score

from brainturtle import config as cfg, evaluate, featuresets as fs
from brainturtle import report, resampling as rs

report.run_banner("b_removal_sweep")

feats = fs.common_features()
grids = {
    "pre-specified": cfg.REMOVAL_GRID_PRESPECIFIED,
    "exploratory": cfg.REMOVAL_GRID_EXPLORATORY,
}
all_thresholds = sorted(set(cfg.REMOVAL_GRID_PRESPECIFIED)
                        | set(cfg.REMOVAL_GRID_EXPLORATORY))

preds = {}
for pct in all_thresholds:
    keep = fs.keep_set(pct, feats)
    y, p, _ = evaluate.pooled_predictions(keep)
    preds[pct] = (y, p, len(keep))

y0, p0, _ = preds[0.0]

rows = []
for pct in all_thresholds:
    y, p, n_keep = preds[pct]
    mean, lo, hi, _ = rs.bootstrap_auc(y, p)
    row = {
        "drop_pct": pct,
        "in_prespecified": pct in cfg.REMOVAL_GRID_PRESPECIFIED,
        "in_exploratory": pct in cfg.REMOVAL_GRID_EXPLORATORY,
        "n_features_kept": n_keep,
        "auc_point": roc_auc_score(y, p),
        "auc_boot_mean": mean, "ci_low": lo, "ci_high": hi,
    }
    if pct == 0.0:
        row.update({"delta_vs_baseline": 0.0, "p_wald": float("nan"),
                    "p_percentile": float("nan"), "stars": "-"})
    else:
        d = rs.bootstrap_auc_difference(y0, p, p0)
        floor = d["p_percentile_floor"]
        row.update({
            "delta_vs_baseline": d["mean_diff"],
            "diff_ci_low": d["ci_low"], "diff_ci_high": d["ci_high"],
            "z": d["z"], "p_wald": d["p_wald"],
            "p_percentile": d["p_percentile"],
            "p_percentile_floor": floor,
            "p_percentile_reported": (f"< {floor:.2g}"
                                      if d["p_percentile"] < floor
                                      else f"{d['p_percentile']:.4g}"),
            "stars": rs.stars(d["p_wald"]),
        })
    rows.append(row)

res = pd.DataFrame(rows)

report.header("Figure 2a / Supplementary S1 - removal ladder",
              f"N_BOOT={cfg.N_BOOT}, SEEDS={cfg.SEEDS}")
report.show(res, ["drop_pct", "in_prespecified", "in_exploratory",
                  "n_features_kept", "auc_point", "auc_boot_mean",
                  "ci_low", "ci_high"])

print()
print("  AUC-difference test vs 0% baseline")
print("  (paired case-resampling, Wald z from the bootstrap SD --")
print("   NOT the analytic DeLong procedure)")
report.show(res[res.drop_pct > 0],
            ["drop_pct", "delta_vs_baseline", "z", "p_wald",
             "p_percentile_reported", "stars"])

peak = res.loc[res.auc_boot_mean.idxmax()]
print()
print(f"  peak: {peak.drop_pct:.0%} removal, AUC {peak.auc_boot_mean:.4f} "
      f"[{peak.ci_low:.4f}, {peak.ci_high:.4f}]")
pre = res[res.in_prespecified & (res.drop_pct > 0)]
print(f"  pre-specified grid peak: "
      f"{pre.loc[pre.auc_boot_mean.idxmax()].drop_pct:.0%}")

report.save(res, "fig2a_removal_ladder")
