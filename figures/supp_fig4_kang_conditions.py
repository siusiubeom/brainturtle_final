"""Kang removal sweep, both recording conditions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figstyle as F
from brainturtle import config as cfg

F.apply()

RES = cfg.RESULTS / f"n{cfg.N_PERM}"
sweep = pd.read_csv(RES / "kang_threshold_sweep.csv")

FULL = "Kang"
CUT = "Kang (investigator-removed)"

SERIES = [
    (FULL, "Full corpus", F.C.AD, "-"),
    (CUT, "Investigator speech removed", F.C.MUTED, "--"),
]

fig, ax = plt.subplots(figsize=(F.COL1 * 1.35, 3.35))

ax.axhline(0.5, color=F.C.ACCENT, lw=0.8, ls=":", zorder=2)
ax.text(-2.0, 0.505, "chance", fontsize=6.0, ha="left", va="bottom",
        color=F.C.ACCENT)

peaks = {}

for cohort, label, colour, ls in SERIES:
    d = sweep[sweep["cohort"] == cohort].sort_values("drop_pct")
    x = d["drop_pct"].values * 100.0
    auc = d["auc"].values
    lo = d["ci_low"].values
    hi = d["ci_high"].values
    sig = d["above_chance"].values.astype(bool)

    ax.fill_between(x, lo, hi, color=colour, alpha=0.16, lw=0, zorder=1)
    ax.plot(x, auc, color=colour, lw=1.5, ls=ls, zorder=4, label=label)
    ax.scatter(x[sig], auc[sig], s=26, color=colour, zorder=5,
               edgecolors="white", linewidths=0.6)
    ax.scatter(x[~sig], auc[~sig], s=22, facecolors="white",
               edgecolors=colour, linewidths=1.0, zorder=5)

    i = int(np.argmax(auc))
    peaks[cohort] = dict(x=x[i], auc=auc[i], p=d["p"].values[i],
                         lo=lo[i], hi=hi[i], colour=colour)

peak_x = peaks[FULL]["x"]
ax.axvline(peak_x, color=F.C.HILITE, lw=0.8, ls="--", alpha=0.6, zorder=2)

for cohort in (FULL, CUT):
    pk = peaks[cohort]
    ax.scatter([pk["x"]], [pk["auc"]], s=58, marker="D", color=F.C.HILITE,
               edgecolors="white", linewidths=0.7, zorder=6)

ax.text(peak_x, 0.875, "both peak at 35%", fontsize=6.5, ha="center",
        va="bottom", color=F.C.HILITE, fontweight="bold")

peak_txt = (
    f"peak AUC at 35% removal\n"
    f"full corpus            {peaks[FULL]['auc']:.4f}  "
    f"(p = {peaks[FULL]['p']:.4f} {F.stars(peaks[FULL]['p'])})\n"
    f"investigator-removed   {peaks[CUT]['auc']:.4f}  "
    f"(p = {peaks[CUT]['p']:.4f} {F.stars(peaks[CUT]['p'])})"
)
ax.text(42.0, 0.278, peak_txt, fontsize=6.2, ha="right", va="bottom",
        color=F.C.HILITE, linespacing=1.45, zorder=7,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor=F.C.HILITE, linewidth=0.5, alpha=0.92))

ax.set_xlabel("Age-sensitive features removed (% of the SHAP ranking)")
ax.set_ylabel("AUC (Kang corpus, speaker-held-out)")
ax.set_title("Kang corpus removal sweep, both recording conditions", pad=7)
ax.set_xticks(np.arange(0, 41, 5))
ax.set_xlim(-2.5, 42.5)
ax.set_ylim(0.26, 0.92)

series_handles, series_labels = ax.get_legend_handles_labels()
fill_handles = [
    plt.Line2D([], [], marker="o", ls="none", ms=4.5, color=F.C.ACCENT,
               markeredgecolor="white", markeredgewidth=0.6),
    plt.Line2D([], [], marker="o", ls="none", ms=4.2, markerfacecolor="white",
               markeredgecolor=F.C.ACCENT, markeredgewidth=1.0),
]
leg = ax.legend(series_handles + fill_handles,
                series_labels + ["above chance (p < 0.05)", "not above chance"],
                loc="upper left", ncol=1, handlelength=1.6, borderpad=0.3,
                labelspacing=0.3, frameon=True, facecolor="white",
                edgecolor="none", framealpha=0.95)
leg.set_zorder(8)

print("[supplementary_fig4_kang_conditions]")
F.save(fig, "supplementary_fig4_kang_conditions")

for cohort, label, _, _ in SERIES:
    pk = peaks[cohort]
    print(f"    {label:30s} peak AUC {pk['auc']:.4f} at {pk['x']:.0f}% "
          f"CI [{pk['lo']:.4f}, {pk['hi']:.4f}]  p = {pk['p']:.4f}")
print(f"    manuscript reports 0.668 (full) and 0.651 (investigator-removed); "
      f"corrected values are higher")
