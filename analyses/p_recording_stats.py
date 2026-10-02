"""Supplementary S11 / S12 — recording duration and speech ratio."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from brainturtle import config as cfg, report

report.run_banner("p_recording_stats")

meta = pd.read_csv(cfg.DATA / "metadata.csv")
COLS = ["speaker_id", "label", "duration_sec", "speech_ratio",
        "speech_sec", "silence_sec"]
missing = [c for c in COLS if c not in meta.columns]
if missing:
    raise SystemExit(f"metadata.csv is missing {missing}")

rec = meta[COLS].dropna(subset=["duration_sec", "speech_ratio"]).copy()
print(f"  {len(rec)} recordings from {rec.speaker_id.nunique()} speakers "
      f"({(rec.label == 1).sum()} AD, {(rec.label == 0).sum()} control)")


def upper_outliers(x):
    """Tukey's rule."""
    q1, q3 = np.percentile(x, [25, 75])
    return int((x > q3 + 1.5 * (q3 - q1)).sum())


def compare(metric):
    ctrl = rec.loc[rec.label == 0, metric].values
    ad = rec.loc[rec.label == 1, metric].values

    u, p_mw = stats.mannwhitneyu(ad, ctrl, alternative="two-sided")
    rbc = 2 * u / (len(ad) * len(ctrl)) - 1
    _, p_lev = stats.levene(ad, ctrl, center="median")

    return {
        "metric": metric,
        "n_control": len(ctrl), "n_ad": len(ad),
        "median_control": float(np.median(ctrl)),
        "median_ad": float(np.median(ad)),
        "sd_control": float(np.std(ctrl, ddof=1)),
        "sd_ad": float(np.std(ad, ddof=1)),
        "upper_outliers_control": upper_outliers(ctrl),
        "upper_outliers_ad": upper_outliers(ad),
        "mannwhitney_u": float(u), "p_location": float(p_mw),
        "rank_biserial": float(rbc),
        "levene_p_spread": float(p_lev),
    }


summary = pd.DataFrame([compare("duration_sec"), compare("speech_ratio")])

report.header("Recording duration and speech ratio (Supplementary S11/S12)",
              f"source: data/metadata.csv, {len(rec)} recordings")
report.show(summary, ["metric", "median_control", "median_ad",
                      "sd_control", "sd_ad", "p_location",
                      "rank_biserial", "levene_p_spread"])
print()
report.show(summary, ["metric", "upper_outliers_control",
                      "upper_outliers_ad"], fmt="{:.0f}")

d = summary.iloc[0]
s = summary.iloc[1]

print()
print("  S11 (duration) -- claim: AD greater median, higher variance, "
      "more upper outliers")
print(f"    median  {d['median_control']:.1f}s -> {d['median_ad']:.1f}s "
      f"({'higher' if d['median_ad'] > d['median_control'] else 'LOWER'}), "
      f"p = {d['p_location']:.4g}")
print(f"    SD      {d['sd_control']:.1f} -> {d['sd_ad']:.1f} "
      f"({'higher' if d['sd_ad'] > d['sd_control'] else 'LOWER'}), "
      f"Levene p = {d['levene_p_spread']:.4g}")
print(f"    upper outliers  {d['upper_outliers_control']:.0f} -> "
      f"{d['upper_outliers_ad']:.0f}")

print()
print("  S12 (speech ratio) -- claim: distributions largely overlapping")
print(f"    median  {s['median_control']:.3f} -> {s['median_ad']:.3f}, "
      f"p = {s['p_location']:.4g}")
print(f"    rank-biserial {s['rank_biserial']:+.3f} "
      f"(0 = complete overlap)")
if s["p_location"] < cfg.ALPHA:
    print("    NOTE: the distributions overlap heavily but differ "
          "significantly in location.")
    print("    'largely overlapping' is fair as a description of the "
          "spread; it should not be")
    print("    read as 'comparable', which the current legend implies.")

report.save(rec, "recording_stats")
report.save(summary, "recording_stats_summary")
