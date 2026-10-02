"""Curve coordinates for the ROC and calibration panels."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

from brainturtle import config as cfg, data, evaluate, featuresets as fs
from brainturtle import report

report.run_banner("o_curves")

feats = fs.common_features()

preds = {}
for pct in (0.0, cfg.PRIMARY_THRESHOLD):
    y, p, _ = evaluate.pooled_predictions(fs.keep_set(pct, feats))
    preds[pct] = (np.asarray(y), np.asarray(p))

roc_rows = []


def add_roc(cohort, y, p):
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    for f, t in zip(fpr, tpr):
        roc_rows.append({"cohort": cohort, "auc": auc,
                         "fpr": float(f), "tpr": float(t)})
    print(f"  {cohort:32s} n={len(y):4d}  AUC={auc:.4f}  "
          f"{len(fpr)} points")
    return auc


y30, p30 = preds[cfg.PRIMARY_THRESHOLD]
add_roc("Pitt (30% removal, pooled OOF)", y30, p30)

for name, _ in data.external_cohorts():
    df = data.external(name)
    P, y = evaluate.external_probability_matrix(df, cfg.PRIMARY_THRESHOLD)
    add_roc(name, y, P.mean(axis=1))

roc = pd.DataFrame(roc_rows)

PINNED_BINS = 10
PINNED_SCHEME = "equal-width"


def reliability(y, p, n_bins=PINNED_BINS, scheme=PINNED_SCHEME):
    """Bin centres, observed frequency and mean predicted probability."""
    if scheme == "equal-width":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        edges = np.unique(edges)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        out.append({"bin_low": float(lo), "bin_high": float(hi),
                    "n": int(m.sum()), "mean_predicted": float(p[m].mean()),
                    "observed_frequency": float(y[m].mean())})
    return out


def ece_from_bins(y, p, n_bins, scheme):
    total, n = 0.0, len(y)
    for b in reliability(y, p, n_bins, scheme):
        total += (b["n"] / n) * abs(b["observed_frequency"]
                                    - b["mean_predicted"])
    return float(total)


cal_rows = []
for pct, (y, p) in preds.items():
    for b in reliability(y, p):
        cal_rows.append({"model": f"{pct:.0%} removal", **b})
cal = pd.DataFrame(cal_rows)

ece_rows = []
for scheme in ("equal-width", "equal-frequency"):
    for n_bins in (5, 10, 15, 20):
        row = {"scheme": scheme, "n_bins": n_bins}
        for pct, (y, p) in preds.items():
            row[f"ece_{pct:.0%}"] = ece_from_bins(y, p, n_bins, scheme)
        ece_rows.append(row)
ece = pd.DataFrame(ece_rows)

report.header("Calibration curves and ECE conventions (Supplementary S5)",
              f"pinned convention: {PINNED_BINS} {PINNED_SCHEME} bins")

for pct, (y, p) in preds.items():
    print(f"  {pct:.0%} removal: AUC={roc_auc_score(y, p):.4f}  "
          f"Brier={brier_score_loss(y, p):.4f}  "
          f"ECE={ece_from_bins(y, p, PINNED_BINS, PINNED_SCHEME):.4f}")

print()
print("  ECE under every plausible convention:")
report.show(ece, fmt="{:.4f}")

MS_ECE = {"ece_0%": 0.1714, "ece_30%": 0.1366}
print()
print(f"  manuscript reports {MS_ECE['ece_0%']:.4f} / {MS_ECE['ece_30%']:.4f}")
TOL = 0.002
close = ece[(ece["ece_0%"] - MS_ECE["ece_0%"]).abs().lt(TOL)
            & (ece["ece_30%"] - MS_ECE["ece_30%"]).abs().lt(TOL)]
if len(close):
    print("  reproduced by:")
    report.show(close, fmt="{:.4f}")
else:
    print("  NO convention in this sweep reproduces both manuscript values.")
    print("  The reported ECE cannot be traced to a binning scheme; use the")
    print("  pinned convention above and state it in Methods.")

report.save(roc, "roc_curves")
report.save(cal, "calibration_bins")
report.save(ece, "ece_conventions")
