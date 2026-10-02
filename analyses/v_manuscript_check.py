"""Check the manuscript's printed numbers against the result tables."""
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd
from lxml import etree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from brainturtle import config as cfg

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
RES = cfg.RESULTS / f"n{cfg.N_PERM}"
DEFAULT_DOC = Path(__file__).resolve().parent.parent / "npj_DM_MS_final.docx"


def manuscript_text(path):
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    out = []
    for p in root.iter(W + "p"):
        out.append("".join(t.text or "" for t in p.iter(W + "t")))
    return "\n".join(out)


def cell(csv, where, column):
    """One value from a result table. `where` is (column, value) or None."""
    df = pd.read_csv(RES / csv)
    if where is not None:
        col, val = where
        df = df[df[col].astype(str) == str(val)]
        if df.empty:
            raise KeyError("%s: no row where %s == %r" % (csv, col, val))
    return float(df.iloc[0][column])


CHECKS = [
    ("Pitt age gap", "7.09", "fig1_age_benchmark.csv",
     ("quantity", "age gap AD - control (years)"), "value", 2),
    ("Pitt age-only AUC", "0.714", "fig1_age_benchmark.csv",
     ("quantity", "AUC, age alone"), "value", 3),
    ("Pitt 0% AUC", "0.698", "fig1_age_benchmark.csv",
     ("quantity", "AUC, 0% removal (full feature space)"), "value", 3),
    ("Pitt 30% AUC (P2)", "0.756", "fig1_age_benchmark.csv",
     ("quantity", "AUC, 30% removal"), "value", 3),
    ("0% vs age difference", "-0.016", "fig1_age_benchmark.csv",
     ("quantity", "AUC difference, 0% model vs age"), "value", 3),
    ("30% vs age difference", "+0.0424", "fig1_age_benchmark.csv",
     ("quantity", "AUC difference, 30% model vs age"), "value", 4),

    ("Pitt baseline (P1)", "0.684", "fig2a_removal_ladder.csv",
     ("drop_pct", "0.0"), "auc_point", 3),
    ("Pitt peak (P1)", "0.740", "fig2a_removal_ladder.csv",
     ("drop_pct", "0.3"), "auc_point", 3),
    ("targeted vs random dAUC", "0.056", "fig2b_summary.csv", None,
     "observed", 3),
    ("targeted vs random p", "0.0012", "fig2b_summary.csv", None, "p", 4),

    ("incremental age alone", "0.709", "fig4_incremental.csv",
     ("model", "age alone"), "auc", 3),
    ("incremental undeconfounded", "0.062", "fig4_incremental.csv",
     ("model", "age + undeconfounded score"), "increment", 3),
    ("incremental deconfounded", "0.104", "fig4_incremental.csv",
     ("model", "age + deconfounded score"), "increment", 3),
    ("difference of increments", "+0.0419", "fig4_incremental.csv",
     ("model", "difference of increments (deconfounded - undeconfounded)"),
     "increment", 4),
    ("difference of increments p", "0.004", "fig4_incremental.csv",
     ("model", "difference of increments (deconfounded - undeconfounded)"),
     "p", 3),

    ("Kang age gap", "6.42", "kang_age_benchmark.csv",
     ("quantity", "age gap MCI - HC (years)"), "value", 2),
    ("Kang age-only AUC", "0.840", "kang_age_benchmark.csv",
     ("quantity", "AUC, age alone"), "value", 3),
    ("Kang 0% AUC", "0.455", "kang_threshold_sweep.csv", None, None, 3),
    ("Kang 30% AUC", "0.660", "kang_threshold_sweep.csv", None, None, 3),
    ("Kang 35% AUC", "0.732", "kang_threshold_sweep.csv", None, None, 3),
    ("Kang 0% vs age", "0.385", "kang_age_benchmark.csv",
     ("quantity", "AUC difference, 0% removal (undeconfounded) vs age"),
     "value", 3),
    ("Kang 30% vs age", "0.179", "kang_age_benchmark.csv",
     ("quantity", "AUC difference, 30% removal vs age"), "value", 3),
    ("Kang incremental 35%", "+0.0689", "fig4_kang_incremental.csv",
     ("model", "age + 35% removal score"), "increment", 4),
    ("Kang incremental undeconf", "+0.0007", "fig4_kang_incremental.csv",
     ("model", "age + 0% removal (undeconfounded) score"), "increment", 4),
    ("Kang difference of increments", "+0.0682", "fig4_kang_incremental.csv",
     ("model", "difference of increments (deconfounded - undeconfounded)"),
     "increment", 4),
    ("Kang 35 vs 30, cleaned", "+0.130", "kang_threshold_contrast.csv",
     ("cohort", "Kang (investigator-removed)"), "diff", 3),
    ("Kang 35 vs 30, full", "+0.072", "kang_threshold_contrast.csv",
     ("cohort", "Kang"), "diff", 3),

    ("Brier 0%", "0.240", "calibration.csv", ("model", "0% removal"),
     "brier", 3),
    ("Brier 30%", "0.209", "calibration.csv", ("model", "30% removal"),
     "brier", 3),
    ("ECE 0%", "0.171", "calibration.csv", ("model", "0% removal"), "ece", 3),
    ("ECE 30%", "0.137", "calibration.csv", ("model", "30% removal"), "ece", 3),

    ("Pitt sensitivity", "0.59", "operating_point.csv",
     ("cohort", "Pitt (threshold chosen here)"), "sensitivity", 2),
    ("Pitt specificity", "0.80", "operating_point.csv",
     ("cohort", "Pitt (threshold chosen here)"), "specificity", 2),
    ("Kang sensitivity", "0.35", "operating_point.csv", ("cohort", "Kang"),
     "sensitivity", 2),
    ("Kang specificity", "0.81", "operating_point.csv", ("cohort", "Kang"),
     "specificity", 2),
    ("ADReSSo AUC", "0.859", "fig4_external.csv", ("cohort", "ADReSSo-2021"),
     "auc", 3),

    ("MFCC removed", "27", "fig3_category_rates.csv", ("category", "MFCC"),
     "dropped", 0),
    ("MFCC shape removed", "15", "fig3_mfcc_subfamily.csv",
     ("subfamily", "shape"), "dropped", 0),

    ("HDBSCAN clusters", "no discrete", "geometry_summary.csv",
     ("space", "deconfounded (77 features)"), "hdbscan_clusters", 0),

    ("suppression k=44 dAUC", "0.045", "fig1_suppression.csv",
     ("k", "44"), "delta", 3),
    ("suppression k=44 p", "0.0062", "fig1_suppression.csv",
     ("k", "44"), "p", 4),
    ("suppression k=5 dAUC", "0.066", "fig1_suppression.csv",
     ("k", "5"), "delta", 3),
    ("suppression k=5 p", "0.091", "fig1_suppression.csv", ("k", "5"), "p", 3),
    ("suppression k=11 p", "0.568", "fig1_suppression.csv",
     ("k", "11"), "p", 3),

    ("gender observed dAUC", "0.0132", "suppfig9_summary.csv", None,
     "observed", 4),
    ("gender null low", "0.042", "suppfig9_summary.csv", None, "null_lo", 3),
    ("gender null high", "0.036", "suppfig9_summary.csv", None, "null_hi", 3),

    ("ranking rho, partition", "0.58", "age_model_partition.csv", None,
     "spearman_rank_rho", 2),
    ("ranking overlap top-11", "8", "age_model_partition.csv", None,
     "overlap_top11", 0),
    ("ranking overlap top-33", "19", "age_model_partition.csv", None,
     "overlap_top33", 0),

    ("duration median control", "58.51", "recording_stats_summary.csv",
     ("metric", "duration_sec"), "median_control", 2),
    ("duration median AD", "69.89", "recording_stats_summary.csv",
     ("metric", "duration_sec"), "median_ad", 2),
    ("duration rank-biserial", "0.222", "recording_stats_summary.csv",
     ("metric", "duration_sec"), "rank_biserial", 3),

    ("Pitt speakers", "292", "table1_cohorts.csv",
     ("cohort", "DementiaBank Pitt"), "n_speakers", 0),
    ("Common Voice speakers", "417", "table1_cohorts.csv",
     ("cohort", "Mozilla Common Voice"), "n_speakers", 0),

    ("detector k=44 dAUC", "0.031", "item8_detector_robustness.csv", None,
     "delta", 3),
    ("detector k=44 p", "0.089", "item8_detector_robustness.csv", None,
     "p", 3),

    ("removed |rho| all", "0.071", "feature_age_corr.csv",
     ("group", "removed at 30%"), "mean_abs_rho", 3),
    ("retained |rho| all", "0.076", "feature_age_corr.csv",
     ("group", "retained at 30%"), "mean_abs_rho", 3),
]

DETECTOR = ("SHAP (canonical)", "IsolationForest", 44)

SWEEP = {"Kang 0% AUC": 0.0, "Kang 30% AUC": 0.30, "Kang 35% AUC": 0.35}


def main(doc):
    text = manuscript_text(doc)
    fails, missing = [], []
    print("checking %d values against %s\n" % (len(CHECKS), RES))
    for label, printed, csv, where, column, dec in CHECKS:
        if label.startswith("detector k=44"):
            df = pd.read_csv(RES / csv)
            r, d, k = DETECTOR
            row = df[(df.ranking == r) & (df.detector == d) & (df.k == k)]
            got = float(row.iloc[0][column])
        elif label in SWEEP:
            df = pd.read_csv(RES / csv)
            row = df[(df.cohort == "Kang") & (df.drop_pct == SWEEP[label])]
            got = float(row.iloc[0]["auc"])
        else:
            try:
                got = cell(csv, where, column)
            except (FileNotFoundError, KeyError) as e:
                print("  MISSING  %-32s %s" % (label, e))
                missing.append(label)
                continue

        want = printed.lstrip("+")
        try:
            want_f = float(want)
        except ValueError:
            want_f = None

        in_text = printed.lstrip("+") in text or printed in text
        ok = True
        if want_f is not None:
            ok = round(abs(got), dec) == round(abs(want_f), dec)
        flag = "ok   " if ok else "WRONG"
        if not ok:
            fails.append((label, printed, got, csv))
        print("  %s %-32s manuscript %-9s table %-12s %s"
              % (flag, label, printed, ("%%.%df" % dec) % got,
                 "" if in_text else "  [not found in text]"))

    print()
    if missing:
        print("%d table(s) missing -- rerun incomplete" % len(missing))
    if fails:
        print("%d MISMATCH(ES):" % len(fails))
        for label, printed, got, csv in fails:
            print("   %-32s manuscript %-10s regenerated %.6f   (%s)"
                  % (label, printed, got, csv))
    else:
        print("every checked value matches the regenerated tables")
    return 1 if fails else 0


if __name__ == "__main__":
    doc = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DOC
    sys.exit(main(doc))
