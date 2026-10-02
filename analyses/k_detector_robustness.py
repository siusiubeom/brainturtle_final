"""Is the suppression pattern specific to Isolation Forest?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from brainturtle import config as cfg, data, evaluate, featuresets as fs
from brainturtle import report, resampling as rs

report.run_banner("k_detector_robustness")

pitt = data.pitt()
ALL = [c for c in fs.common_features()
       if c not in {"speaker_id", "label", "split", "speaker_int"}]

K_VALUES = [int(x) for x in (5, 11, 22, 33, 44)]


def _split_arrays(seed, keep):
    trm, vam = evaluate.split_masks(seed)
    return (pitt.loc[trm, keep].values,
            pitt.loc[vam, keep].values,
            pitt.loc[vam, "label"].values)


def score_isoforest(Xtr, Xva):
    m = IsolationForest(random_state=0, contamination="auto",
                        n_jobs=cfg.MODEL_THREADS).fit(Xtr)
    return -m.decision_function(Xva)


def score_lof(Xtr, Xva):
    m = LocalOutlierFactor(n_neighbors=20, novelty=True,
                           n_jobs=cfg.MODEL_THREADS).fit(Xtr)
    return -m.decision_function(Xva)


def score_mahalanobis(Xtr, Xva):
    s = StandardScaler().fit(Xtr)
    cov = EmpiricalCovariance().fit(s.transform(Xtr))
    return cov.mahalanobis(s.transform(Xva))


def score_pca_recon(Xtr, Xva):
    s = StandardScaler().fit(Xtr)
    Ztr, Zva = s.transform(Xtr), s.transform(Xva)
    n = max(1, min(Ztr.shape[1] - 1, int(Ztr.shape[1] * 0.5)))
    p = PCA(n_components=n, random_state=0).fit(Ztr)
    return ((Zva - p.inverse_transform(p.transform(Zva))) ** 2).sum(axis=1)


DETECTORS = {
    "IsolationForest": score_isoforest,
    "LOF": score_lof,
    "Mahalanobis": score_mahalanobis,
    "PCA-reconstruction": score_pca_recon,
}


def make_scorer(fn):
    def pooled(keep):
        ys, ps = [], []
        for seed in cfg.SEEDS:
            Xtr, Xva, y = _split_arrays(seed, list(keep))
            ys.append(y)
            ps.append(fn(Xtr, Xva))
        return roc_auc_score(np.concatenate(ys), np.concatenate(ps))
    return pooled


RANKINGS = {
    "SHAP (canonical)": fs.CANONICAL_RANKING,
    "gain (alternate)": fs.ALTERNATE_RANKING,
}

rows = []
for rname, rfile in RANKINGS.items():
    ranked = [f for f in fs.load_shap_ranking(rfile)["feature"] if f in set(ALL)]
    for dname, fn in DETECTORS.items():
        scorer = make_scorer(fn)
        for k in K_VALUES:
            observed = scorer(ranked[:k])
            null = rs.subset_permutation_null(scorer, ALL, k)
            s = rs.summarize_null(null, observed, two_sided=True)
            rows.append({
                "ranking": rname, "detector": dname, "k": k,
                "age_auc": observed, "null_mean": s["null_mean"],
                "null_sd": s["null_sd"], "delta": observed - s["null_mean"],
                "z": s["z"], "p": s["p"],
                "worse_than_random": observed < s["null_mean"],
                "significant": s["p"] < cfg.ALPHA,
            })
            print(f"  {rname:18s} {dname:20s} k={k:3d}  "
                  f"age={observed:.4f} null={s['null_mean']:.4f} "
                  f"z={s['z']:+6.2f}  p={s['p']:.4f}")

res = pd.DataFrame(rows)

report.header("Item 8 - suppression across detectors and rankings",
              f"N_PERM={cfg.N_PERM}, SEEDS={cfg.SEEDS}")
for rname in RANKINGS:
    sub = res[res.ranking == rname]
    print(f"\n  {rname}")
    report.show(sub.pivot(index="k", columns="detector",
                          values="delta").reset_index())
    print("   p:")
    report.show(sub.pivot(index="k", columns="detector",
                          values="p").reset_index())

print()
sup = res[res.worse_than_random & res.significant]
if len(sup):
    print("  significant SUPPRESSION (age-ranked worse than random):")
    report.show(sup, ["ranking", "detector", "k", "delta", "p"])
else:
    print("  no detector/ranking/k combination shows significant suppression")
enh = res[~res.worse_than_random & res.significant]
if len(enh):
    print("\n  significant ENHANCEMENT (age-ranked better than random):")
    report.show(enh, ["ranking", "detector", "k", "delta", "p"])

print()
print("  agreement across detectors on the SIGN of the effect, per (ranking,k):")
agree = (res.groupby(["ranking", "k"])["worse_than_random"]
         .agg(["sum", "count"]).reset_index()
         .rename(columns={"sum": "detectors_showing_suppression",
                          "count": "detectors"}))
report.show(agree)

report.save(res, "item8_detector_robustness")
