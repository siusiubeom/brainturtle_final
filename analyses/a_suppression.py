"""Figure 1b/1c — do age-ranked features actively suppress the disease signal?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from brainturtle import config as cfg, data, evaluate, featuresets as fs
from brainturtle import report, resampling as rs

report.run_banner("a_suppression")

pitt = data.pitt()
all_feats = [c for c in fs.common_features()
             if c not in {"speaker_id", "label", "split", "speaker_int"}]
ranking = fs.load_shap_ranking()
age_rank = [f for f in ranking["feature"] if f in set(all_feats)]
print(f"  {len(all_feats)} features, age ranking covers {len(age_rank)}")

rows = []
for pct in cfg.KEEP_PERCENTS:
    k = max(1, int(len(age_rank) * pct))
    observed = evaluate.isolation_forest_auc(age_rank[:k])

    def make_null(seed, n_perm, _k=k):
        return rs.subset_permutation_null(evaluate.isolation_forest_auc,
                                          all_feats, _k,
                                          n_perm=n_perm, seed=seed)

    null = make_null(cfg.PERM_SEED, cfg.N_PERM)
    s = rs.summarize_null(null, observed, two_sided=True)
    rows.append({"k": k, "pct": pct, "age_auc": observed,
                 "null_mean": s["null_mean"], "null_sd": s["null_sd"],
                 "null_lo": s["null_lo"], "null_hi": s["null_hi"],
                 "delta": observed - s["null_mean"], "z": s["z"],
                 "p": s["p"], "p_formatted": s["p_formatted"],
                 "n_ge_observed": s["n_ge_observed"],
                 "worse_than_random": observed < s["null_mean"]})
    print(f"    k={k:3d}  age={observed:.4f}  null={s['null_mean']:.4f}  "
          f"z={s['z']:+6.2f}  {s['p_formatted']}")

res = pd.DataFrame(rows)

report.header("Figure 1b/1c - age-ranked vs random feature subsets",
              f"Isolation Forest, N_PERM={cfg.N_PERM}, SEEDS={cfg.SEEDS}")
report.show(res, ["k", "pct", "age_auc", "null_mean", "null_lo", "null_hi",
                  "delta", "z", "p"])

sig = res[(res.p < cfg.ALPHA) & res.worse_than_random]
print()
if len(sig):
    print("  feature counts where age-ranked is significantly WORSE than random:")
    report.show(sig, ["k", "age_auc", "null_mean", "delta", "p"])
else:
    print("  no feature count shows significant suppression at "
          f"p<{cfg.ALPHA}")
better = res[(res.p < cfg.ALPHA) & ~res.worse_than_random]
if len(better):
    print("  feature counts where age-ranked is significantly BETTER:")
    report.show(better, ["k", "age_auc", "null_mean", "delta", "p"])

report.save(res, "fig1_suppression")
