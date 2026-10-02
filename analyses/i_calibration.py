"""Calibration and Brier skill."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

from brainturtle import config as cfg, data, evaluate, featuresets as fs
from brainturtle import report, resampling as rs

report.run_banner("i_calibration")

feats = fs.common_features()


def ece(y, p, n_bins=10):
    """Expected calibration error, equal-width bins."""
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


rows = []
preds = {}
for pct in (0.0, cfg.PRIMARY_THRESHOLD):
    y, p, _ = evaluate.pooled_predictions(fs.keep_set(pct, feats))
    preds[pct] = (y, p)
    rows.append({
        "model": f"{pct:.0%} removal",
        "auc": roc_auc_score(y, p),
        "brier": brier_score_loss(y, p),
        "ece": ece(y, p),
    })

res = pd.DataFrame(rows)
bal = data.class_balance(preds[0.0][0])
ref = bal["brier_uninformative"]
res["brier_skill_correct_ref"] = 1 - res["brier"] / ref
res["brier_skill_naive_ref_0.25"] = 1 - res["brier"] / 0.25

report.header("Calibration (Supplementary S5)",
              f"Pitt base rate {bal['base_rate']:.4f}; uninformative Brier "
              f"reference p(1-p) = {ref:.4f}")
report.show(res, ["model", "auc", "brier", "ece",
                  "brier_skill_correct_ref", "brier_skill_naive_ref_0.25"])

b0 = res.loc[0, "brier_skill_correct_ref"]
b30 = res.loc[1, "brier_skill_correct_ref"]
print()
print(f"  Brier skill against the correct reference: "
      f"{b0:+.3f} -> {b30:+.3f}")
print(f"  Brier skill against the naive 0.25 reference: "
      f"{res.loc[0, 'brier_skill_naive_ref_0.25']:+.3f} -> "
      f"{res.loc[1, 'brier_skill_naive_ref_0.25']:+.3f}")
if b0 < 0:
    print()
    print("  NOTE: the 0% baseline has NEGATIVE Brier skill against the")
    print("  correct reference -- its probability estimates are worse than")
    print("  always predicting the base rate. The proposed sentence "
          "'from 3.8% to 16.4%'")
    print("  uses the 0.25 reference and does not survive the correction.")

y30, p30 = preds[cfg.PRIMARY_THRESHOLD]
fpr, tpr, thr = roc_curve(y30, p30)
cut = float(thr[np.argmax(tpr - fpr)])


def op_metrics(y, p, cutoff):
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= cutoff).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "sensitivity": sens, "specificity": spec,
            "ppv": tp / (tp + fp) if tp + fp else np.nan,
            "npv": tn / (tn + fn) if tn + fn else np.nan}


op = [{"cohort": "Pitt (threshold chosen here)", "n": len(y30),
       **op_metrics(y30, p30, cut)}]
for name, _ in data.external_cohorts():
    df = data.external(name)
    P, y = evaluate.external_probability_matrix(df, cfg.PRIMARY_THRESHOLD)
    op.append({"cohort": name, "n": len(y),
               **op_metrics(y, P.mean(axis=1), cut)})
op = pd.DataFrame(op)

report.header("Fixed operating point",
              f"Youden's J on Pitt validation, cutoff = {cut:.4f}, "
              "applied unchanged to every external cohort")
report.show(op, ["cohort", "n", "sensitivity", "specificity", "ppv", "npv"])

sens = op.loc[0, "sensitivity"]
spec = op.loc[0, "specificity"]
prev_rows = []
for prev in (0.02, 0.05, 0.10, 0.20):
    ppv = sens * prev / (sens * prev + (1 - spec) * (1 - prev))
    npv = spec * (1 - prev) / ((1 - sens) * prev + spec * (1 - prev))
    prev_rows.append({"prevalence": prev, "ppv": ppv, "npv": npv})
prev_df = pd.DataFrame(prev_rows)
print()
print("  Predictive values at plausible community prevalences "
      f"(sens={sens:.3f}, spec={spec:.3f}):")
report.show(prev_df)

report.save(res, "calibration")
report.save(op, "operating_point")
report.save(prev_df, "operating_point_prevalence")
report.save([{"youden_cutoff": cut, **bal}], "operating_point_config")
