"""Figure 2c — the removal profile within age-homogeneous strata."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from brainturtle import config as cfg, evaluate, featuresets as fs
from brainturtle import report, resampling as rs

report.run_banner("d_agebins")

feats = fs.common_features()
rows = []

for pct in cfg.REMOVAL_GRID_PRESPECIFIED:
    keep = fs.keep_set(pct, feats)
    _, _, frame = evaluate.pooled_predictions(keep, with_age=True)

    for lo, hi in cfg.AGE_BINS:
        sub = frame[(frame.entryage >= lo) & (frame.entryage < hi)]
        y = sub["label"].to_numpy()
        p = sub["prob"].to_numpy()
        rec = {"drop_pct": pct, "age_bin": f"{lo}-{hi}", "n": len(sub)}
        if len(sub) < 10 or len(np.unique(y)) < 2:
            rec.update({"auc": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                        "null_mean": np.nan, "p": np.nan})
        else:
            auc = roc_auc_score(y, p)
            _, cl, ch, _ = rs.bootstrap_auc(y, p)
            null = rs.label_permutation_null(y, p)
            s = rs.summarize_null(null, auc, two_sided=False)
            rec.update({"auc": auc, "ci_low": cl, "ci_high": ch,
                        "null_mean": s["null_mean"],
                        "n_ge_observed": s["n_ge_observed"],
                        "p": s["p"], "p_formatted": s["p_formatted"]})
        rows.append(rec)

res = pd.DataFrame(rows)

report.header("Figure 2c - per-age-bin performance",
              f"within-bin label permutation, N_PERM={cfg.N_PERM}")
report.show(res, ["drop_pct", "age_bin", "n", "auc", "ci_low", "ci_high",
                  "null_mean", "p"])

ns = res[(res.p >= cfg.ALPHA) & res.p.notna()]
print()
if len(ns):
    print("  strata NOT above chance at p<0.05:")
    report.show(ns, ["drop_pct", "age_bin", "n", "auc", "p"])
else:
    print("  every stratum with usable n beats its within-bin null at p<0.05")

peaks = (res.dropna(subset=["auc"])
            .loc[lambda d: d.groupby("age_bin")["auc"].idxmax()]
            [["age_bin", "drop_pct", "auc"]])
print()
print("  peak removal threshold per stratum:")
report.show(peaks)

report.save(res, "fig2c_agebins")
