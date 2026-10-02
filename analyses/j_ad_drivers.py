"""What drives AD classification after the 30% cut."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

from brainturtle import config as cfg, data, evaluate, featuresets as fs, report

report.run_banner("j_ad_drivers")

feats = fs.common_features()
ranking = fs.load_shap_ranking()
age_rank = {f: i + 1 for i, f in enumerate(ranking["feature"])}

pitt = data.pitt()
keep30 = fs.keep_set(cfg.PRIMARY_THRESHOLD, feats)
fs.assert_matches_models(keep30)


def pooled_shap(pct):
    """SHAP on each seed's held-out split, pooled. Positive = toward AD."""
    pct_int = int(round(pct * 100))
    base = joblib.load(cfg.FEAT_DIR / f"pitt_drop{pct_int}_seed0_feats.pkl")
    sv_all, per_seed = [], []
    for seed in cfg.SEEDS:
        clf = joblib.load(cfg.MODEL_DIR / f"pitt_drop{pct_int}_seed{seed}.pkl")
        f_s = joblib.load(cfg.FEAT_DIR / f"pitt_drop{pct_int}_seed{seed}_feats.pkl")
        assert list(f_s) == list(base), f"feature order differs at seed {seed}"
        _, vam = evaluate.split_masks(seed)
        X = pitt.loc[vam, list(base)].values
        sv = shap.TreeExplainer(clf).shap_values(X)
        if isinstance(sv, list):
            sv = sv[1]
        elif sv.ndim == 3:
            sv = sv[:, :, 1]
        sv_all.append(sv)
        per_seed.append(np.abs(sv).mean(axis=0))
    return np.vstack(sv_all), list(base), np.array(per_seed)


sv30, f30, per_seed = pooled_shap(cfg.PRIMARY_THRESHOLD)
drivers = pd.DataFrame({
    "feature": f30,
    "mean_abs_shap": np.abs(sv30).mean(axis=0),
    "seed_sd": per_seed.std(axis=0, ddof=1),
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
drivers["ad_rank"] = drivers.index + 1
drivers["category"] = drivers["feature"].apply(fs.categorize)
drivers["mfcc_subfamily"] = drivers["feature"].apply(fs.mfcc_subfamily)
drivers["age_rank"] = drivers["feature"].map(age_rank)
drivers["share_pct"] = 100 * drivers["mean_abs_shap"] / drivers["mean_abs_shap"].sum()
drivers["cv_pct"] = 100 * drivers["seed_sd"] / drivers["mean_abs_shap"]

report.header("1. Top AD drivers after the 30% cut",
              f"{len(f30)} surviving features, pooled held-out n={sv30.shape[0]}")
report.show(drivers.head(20), ["ad_rank", "feature", "mean_abs_shap", "seed_sd",
                               "cv_pct", "share_pct", "category", "age_rank"])

cat = (drivers.groupby("category")
       .agg(n=("feature", "count"), total=("mean_abs_shap", "sum"))
       .reset_index())
cat["share_pct"] = 100 * cat["total"] / cat["total"].sum()
cat["per_feature"] = cat["total"] / cat["n"]
cat = cat.sort_values("per_feature", ascending=False)
report.header("2. Where the signal sits, by family",
              "share_pct reflects how MANY features; per_feature is density")
report.show(cat, ["category", "n", "share_pct", "per_feature"])

mf = drivers[drivers.category == "MFCC"]
sub = (mf.groupby("mfcc_subfamily")
       .agg(n=("feature", "count"), total=("mean_abs_shap", "sum")).reset_index())
sub["per_feature"] = sub["total"] / sub["n"]
sub["share_of_mfcc_pct"] = 100 * sub["total"] / sub["total"].sum()
print()
print("  MFCC shape vs non-shape (manuscript credits SHAPE for transfer):")
report.show(sub, ["mfcc_subfamily", "n", "share_of_mfcc_pct", "per_feature"])
top20_shape = int((drivers.head(20).mfcc_subfamily == "shape").sum())
print(f"  MFCC shape features in the AD top 20: {top20_shape} of 20")

report.header("3. Inverted-U mechanism",
              "which top AD drivers get deleted at 35% and 40%")
rows = []
for pct in (0.30, 0.35, 0.40):
    dropped = set(fs.drop_set(pct, feats))
    for r in drivers.head(10).itertuples():
        rows.append({"threshold": pct, "feature": r.feature,
                     "ad_rank": r.ad_rank, "age_rank": r.age_rank,
                     "removed": r.feature in dropped})
mech = pd.DataFrame(rows).pivot_table(index=["feature", "ad_rank", "age_rank"],
                                      columns="threshold", values="removed")
mech = mech.reset_index().sort_values("ad_rank")
mech.columns = [str(c) for c in mech.columns]
report.show(mech)
lost = {}
for pct in (0.35, 0.40):
    d = set(fs.drop_set(pct, feats))
    lost[pct] = [r.feature for r in drivers.head(10).itertuples() if r.feature in d]
    print(f"  top-10 AD drivers deleted at {pct:.0%}: {lost[pct] or 'none'}")

report.header("4. Voice quality",
              'Discussion says "shimmer, HNR, were largely removed"')
vq = [f for f in feats if fs.categorize(f) == "Voice Quality"]
dropped30 = set(fs.drop_set(cfg.PRIMARY_THRESHOLD, feats))
for f in vq:
    st = "REMOVED" if f in dropped30 else "retained"
    ad = drivers.loc[drivers.feature == f, "ad_rank"]
    ad_s = f"AD rank {int(ad.iloc[0])}" if len(ad) else "n/a (removed)"
    print(f"  {f:16s} age rank {age_rank[f]:3d}   {st:8s}   {ad_s}")

report.header("5. Cut boundary stability")
vals = ranking.iloc[:, 1].values
k = int(len(feats) * cfg.PRIMARY_THRESHOLD)
gap = vals[k - 1] - vals[k]
rng = vals.max() - vals.min()
print(f"  rank {k}   {ranking.iloc[k-1,0]:16s} {vals[k-1]:.4f}  (in)")
print(f"  rank {k+1} {ranking.iloc[k,0]:16s} {vals[k]:.4f}  (out)")
print(f"  gap {gap:.4f} = {100*gap/rng:.2f}% of the full range {rng:.4f}")
print("  -> individual features near the cut are not robust; this is why the")
print("     paper reports a threshold sweep rather than one selected set.")

report.header("6. SHAP vs Spearman divergence")
cv = data.commonvoice()
age = cv["age_num_x"].values
rho = {f: spearmanr(cv[f].values, age, nan_policy="omit").statistic for f in feats}
sp_rank = {f: i + 1 for i, f in enumerate(
    sorted(feats, key=lambda x: -abs(rho[x])))}
top30_shap = set(list(ranking["feature"])[:30])
top30_sp = set(sorted(feats, key=lambda x: -abs(rho[x]))[:30])
print(f"  top-30 agreement: {len(top30_shap & top30_sp)} of 30")
removed_low_rho = [(f, rho[f]) for f in dropped30 if abs(rho[f]) < 0.03]
kept_high_rho = [(f, rho[f]) for f in feats
                 if f not in dropped30 and abs(rho[f]) >= 0.10]
print(f"  REMOVED with |rho| < 0.03: {len(removed_low_rho)}  "
      f"e.g. {[(f, round(r,3)) for f,r in removed_low_rho[:4]]}")
print(f"  RETAINED with |rho| >= 0.10: {len(kept_high_rho)}  "
      f"e.g. {[(f, round(r,3)) for f,r in kept_high_rho[:4]]}")

drivers["spearman_rho"] = drivers["feature"].map(rho)
drivers["spearman_rank"] = drivers["feature"].map(sp_rank)

report.save(drivers, "ad_drivers")
report.save(cat, "ad_drivers_by_family")
report.save(sub, "ad_drivers_mfcc_subfamily")
report.save(mech, "inverted_u_mechanism")
report.save([{"cut_rank": k, "gap": gap, "range": rng,
              "gap_pct_of_range": 100 * gap / rng,
              "top30_shap_spearman_agreement": len(top30_shap & top30_sp)}],
            "cut_boundary")
