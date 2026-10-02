"""Voice-quality extraction audit, feature-age correlation, converter sensitivity."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brainturtle import config as cfg, data, evaluate, featuresets as fs, report

VQ = ["shimmer_local", "hnr_mean"]

report.run_banner("t_reviewer_checks")


def vq_audit(outdir):
    rows = []
    cohorts = [("DementiaBank Pitt", data.pitt()),
               ("Mozilla Common Voice", data.commonvoice())]
    for name, _ in data.external_cohorts():
        cohorts.append((name, data.external(name)))
    for name, df in cohorts:
        for f in VQ:
            if f not in df.columns:
                continue
            v = pd.to_numeric(df[f], errors="coerce")
            n = len(v)
            counts = v.value_counts(dropna=True)
            top_n = int(counts.iloc[0]) if len(counts) else 0
            rows.append({
                "cohort": name, "feature": f, "n": n,
                "n_missing": int(v.isna().sum()),
                "n_distinct": int(v.nunique()),
                "modal_value_share": top_n / n if n else np.nan,
                "n_at_median": int((v == v.median()).sum()) if v.notna().any() else 0,
                "verdict": ("no missing values, no imputation"
                            if v.isna().sum() == 0 and v.nunique() == n
                            else "check"),
            })
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "vq_extraction_audit.csv", index=False)
    report.header("Voice quality extraction audit", "per corpus")
    report.show(df, ["cohort", "feature", "n", "n_missing", "n_distinct",
                     "modal_value_share", "verdict"])
    return df


def age_correlation(outdir):
    feats = fs.common_features()
    dropped = set(fs.drop_set(cfg.PRIMARY_THRESHOLD, feats))
    kept = [f for f in feats if f not in dropped]
    pitt = data.pitt_with_age().dropna(subset=["entryage"])

    rows = []
    for scope, sub in (("Pitt, all speakers", pitt),
                       ("Pitt, controls only", pitt[pitt["label"] == 0]),
                       ("Pitt, AD only", pitt[pitt["label"] == 1])):
        age = sub["entryage"].values
        for label, group in (("removed at 30%", sorted(dropped)),
                             ("retained at 30%", kept)):
            rhos = []
            for f in group:
                r = spearmanr(sub[f].values, age, nan_policy="omit").statistic
                if np.isfinite(r):
                    rhos.append(abs(r))
            rhos = np.array(rhos)
            rows.append({
                "scope": scope, "group": label, "n_features": len(rhos),
                "mean_abs_rho": float(rhos.mean()),
                "median_abs_rho": float(np.median(rhos)),
                "frac_above_0.10": float((rhos >= 0.10).mean()),
                "frac_above_0.20": float((rhos >= 0.20).mean()),
            })
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "feature_age_corr.csv", index=False)
    report.header("Feature-age correlation, removed vs retained",
                  "the ranking was fitted on Common Voice, never on Pitt")
    report.show(df, ["scope", "group", "n_features", "mean_abs_rho",
                     "frac_above_0.10", "frac_above_0.20"])
    return df


def converter_sensitivity(outdir):
    """Re-score the pooled out-of-fold AUCs without the converting speaker."""
    feats = fs.common_features()
    rows = []
    for pct in (0.0, cfg.PRIMARY_THRESHOLD):
        keep = fs.keep_set(pct, feats)
        y, p, frames = evaluate.pooled_predictions(keep)
        by_speaker = frames.groupby("speaker_id")["label"].nunique()
        converters = sorted(by_speaker[by_speaker > 1].index)
        mask = ~frames["speaker_id"].isin(converters)
        rows.append({
            "threshold": "%d%%" % round(pct * 100),
            "n_converters": len(converters),
            "converter_ids": ";".join(map(str, converters)),
            "n_rows_all": len(frames),
            "auc_all": roc_auc_score(frames["label"], frames["prob"]),
            "n_rows_excl": int(mask.sum()),
            "auc_excl_converter": roc_auc_score(frames.loc[mask, "label"],
                                                frames.loc[mask, "prob"]),
        })
    df = pd.DataFrame(rows)
    df["delta"] = df["auc_excl_converter"] - df["auc_all"]
    df.to_csv(outdir / "converter_sensitivity.csv", index=False)
    report.header("Longitudinal converter sensitivity",
                  "pooled out-of-fold AUC with the converter dropped")
    report.show(df, ["threshold", "n_converters", "n_rows_all", "auc_all",
                     "n_rows_excl", "auc_excl_converter", "delta"])
    return df


def main():
    outdir = cfg.RESULTS / f"n{cfg.N_PERM}"
    outdir.mkdir(parents=True, exist_ok=True)
    vq_audit(outdir)
    age_correlation(outdir)
    converter_sensitivity(outdir)
    print()
    print("  wrote vq_extraction_audit.csv, feature_age_corr.csv, "
          "converter_sensitivity.csv")


if __name__ == "__main__":
    main()
