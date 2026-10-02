"""Refit the Common Voice age model and recompute its SHAP ranking."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

from brainturtle import config as cfg, data, featuresets as fs, report

report.run_banner("q_age_model_shap")

TOP_N = 20
AGE_COL = "age_num_x"

cv = data.commonvoice()
feats = fs.common_features()
X = cv[feats].astype(float)
y = cv[AGE_COL].astype(float)

print(f"  Common Voice: {len(cv)} speakers, {len(feats)} features")
print(f"  age: {y.min():.0f}-{y.max():.0f}, "
      f"{y.nunique()} distinct values -> {sorted(y.unique().astype(int))}")
print("  NOTE: Common Voice records age as a decade band; these are bin")
print("  midpoints, not measured age.")


AGE_MODEL_PARAMS = dict(n_estimators=600, learning_rate=0.03, max_depth=-1,
                        random_state=42)


def fit_and_shap(Xf, yf, Xeval, seed=42, params=None, scale=True,
                 explainer="interventional"):
    """Fit the age regressor and return (mean|SHAP| series, SHAP matrix)."""
    import shap
    from sklearn.preprocessing import StandardScaler

    p = dict(AGE_MODEL_PARAMS if params is None else params)
    p["random_state"] = seed
    if scale:
        sc = StandardScaler().fit(Xf)
        Xf_m = pd.DataFrame(sc.transform(Xf), columns=Xf.columns,
                            index=Xf.index)
        Xe_m = pd.DataFrame(sc.transform(Xeval), columns=Xeval.columns,
                            index=Xeval.index)
    else:
        Xf_m, Xe_m = Xf, Xeval

    m = LGBMRegressor(verbose=-1, n_jobs=cfg.MODEL_THREADS, **p).fit(Xf_m, yf)
    if explainer == "interventional":
        sv = shap.Explainer(m, Xf_m)(Xe_m).values
    else:
        sv = shap.TreeExplainer(m).shap_values(Xe_m)
    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=Xeval.columns)
    return mean_abs.sort_values(ascending=False), sv


def rank_agreement(order, reference):
    """Spearman rho and top-k overlap of one ordering against another."""
    common = [f for f in reference if f in set(order)]
    rho = spearmanr([reference.index(f) for f in common],
                    [list(order).index(f) for f in common]).statistic
    return float(rho), common


Xtr, Xva, ytr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
rank_val, sv_val = fit_and_shap(Xtr, ytr, Xva)
rank_full, sv_full = fit_and_shap(X, y, X)

stored = fs.load_shap_ranking()
stored = stored[stored["feature"].isin(feats)].reset_index(drop=True)
stored_order = list(stored["feature"])

check_rows = []
for name, recomputed in (("validation fit (Fig 1a)", rank_val),
                         ("full-corpus fit (S13)", rank_full)):
    order = list(recomputed.index)
    common = [f for f in stored_order if f in set(order)]
    rho = spearmanr([stored_order.index(f) for f in common],
                    [order.index(f) for f in common]).statistic
    row = {"fit": name, "spearman_rank_rho": float(rho)}
    for k in (5, 11, 33):
        row[f"overlap_top{k}"] = len(set(stored_order[:k]) & set(order[:k]))
    check_rows.append(row)
check = pd.DataFrame(check_rows)

common = list(rank_val.index)
rho_part = spearmanr([list(rank_val.index).index(f) for f in common],
                     [list(rank_full.index).index(f) for f in common]).statistic
part = pd.DataFrame([{
    "comparison": "validation fit vs full-corpus fit",
    "spearman_rank_rho": float(rho_part),
    **{f"overlap_top{k}": len(set(rank_val.index[:k])
                              & set(rank_full.index[:k]))
       for k in (5, 11, 20, 33)},
}])

top = [f for f in stored_order[:TOP_N]]
idx = [list(X.columns).index(f) for f in top]
bee = []
for j, f in zip(idx, top):
    vals = X[f].values
    lo, hi = np.nanpercentile(vals, [1, 99])
    norm = np.clip((vals - lo) / (hi - lo) if hi > lo else vals * 0, 0, 1)
    for i in range(len(X)):
        bee.append({"feature": f, "rank": top.index(f) + 1,
                    "shap": float(sv_full[i, j]),
                    "feature_value_norm": float(norm[i])})
bee = pd.DataFrame(bee)

report.header("Common Voice age-regression SHAP (Figure 1a / S13)",
              f"{len(cv)} speakers; beeswarm over the top {TOP_N} "
              "stored-canonical features")
print("  agreement of the recomputed ranking with the stored canonical file:")
report.show(check, fmt="{:.4f}")
print()
print("  S13 stability claim -- validation partition vs full corpus:")
report.show(part, fmt="{:.4f}")

r = float(part.loc[0, "spearman_rank_rho"])
print()
if r >= 0.9:
    print(f"  rank rho = {r:.3f}: S13's 'stable across partitions' is "
          "supported.")
else:
    print(f"  rank rho = {r:.3f}: the two partitions do NOT agree closely.")
    print("  S13's claim that the ranking is 'stable and not driven by the")
    print("  choice of data partition' overstates what the two fits show.")

print()
print("  The stored ranking is NOT overwritten -- the saved drop30 models")
print("  were fit against it. Any change to it invalidates those pickles.")

report.save(bee, "age_model_shap_values")
report.save(check, "age_model_ranking_check")
report.save(part, "age_model_partition")
