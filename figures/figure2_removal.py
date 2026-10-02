"""Figure 2 — targeted removal of age-sensitive features."""
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
lad = pd.read_csv(RES / "fig2a_removal_ladder.csv")
draws = pd.read_csv(RES / "fig2b_null_draws.csv")
b = pd.read_csv(RES / "fig2b_summary.csv").iloc[0]
bins = pd.read_csv(RES / "fig2c_agebins.csv")

fig = plt.figure(figsize=(F.COL2, 2.9))
gs = fig.add_gridspec(1, 3, wspace=0.75)

ax = fig.add_subplot(gs[0, 0])
pre = lad[lad.in_prespecified].sort_values("drop_pct")
x = pre["drop_pct"] * 100

ax.fill_between(x, pre["ci_low"], pre["ci_high"], color=F.C.CTRL,
                alpha=0.5, lw=0, label="95% CI")
ax.plot(x, pre["auc_point"], color=F.C.AD, lw=1.5, marker="o", ms=4.5,
        label="AUC", zorder=3)

peak = pre.loc[pre["auc_point"].idxmax()]
ax.scatter([peak.drop_pct * 100], [peak.auc_point], s=70, marker="o",
           facecolor="none", edgecolor=F.C.HILITE, lw=1.3, zorder=4)
ax.annotate(f"{peak.auc_point:.3f}\n{peak.stars}",
            xy=(peak.drop_pct * 100, peak.auc_point),
            xytext=(peak.drop_pct * 100 - 1, peak.auc_point + 0.036),
            fontsize=6.4, color=F.C.HILITE, ha="center", va="bottom",
            fontweight="bold")

base = pre[pre.drop_pct == 0].iloc[0]
ax.annotate(f"baseline\n{base.auc_point:.3f}", xy=(0, base.auc_point),
            xytext=(3, base.auc_point - 0.055), fontsize=6,
            color=F.C.ACCENT, ha="left")

ax.set_xlabel("Features removed (%)")
ax.set_ylabel("AUC")
ax.set_title("Removal threshold")
ax.set_xticks([0, 10, 20, 30, 40])
ax.set_ylim(0.60, 0.83)
ax.legend(loc="lower right", handlelength=1.4, frameon=True,
          facecolor="white", edgecolor="none", framealpha=0.95)
F.panel_label(ax, "a")

ax = fig.add_subplot(gs[0, 1])
ax.hist(draws["null_diff"], bins=60, color=F.C.CTRL, edgecolor="white",
        linewidth=0.2, label=f"random 30% removal\n(n = {int(b.n_perm):,})")

ymax = ax.get_ylim()[1]
ax.vlines([b.null_lo, b.null_hi], 0, ymax, color=F.C.MUTED, lw=0.9, ls="--",
          zorder=3, label=f"null 95% [{b.null_lo:+.4f}, {b.null_hi:+.4f}]")
ax.vlines(b.observed, 0, ymax, color=F.C.HILITE, lw=1.8, zorder=4,
          label="targeted removal")

ax.text(b.observed - 0.003, ymax * 1.10,
        f"$\\Delta$AUC = {b.observed:+.4f}\n{b.p_formatted}",
        fontsize=6.4, color=F.C.HILITE, ha="right", va="center",
        fontweight="bold")

ax.set_xlabel("$\\Delta$AUC vs 0% baseline")
ax.set_ylabel("Random removals")
ax.set_title("Targeted vs random removal")
ax.set_ylim(0, ymax * 1.80)
ax.legend(loc="upper left", handlelength=1.2, labelspacing=0.4, fontsize=6.2,
          frameon=True, facecolor="white", edgecolor="none",
          framealpha=0.95, borderpad=0.3)
F.panel_label(ax, "b")

ax = fig.add_subplot(gs[0, 2])
STRATA = [("50-60", F.C.CTRL), ("60-70", F.C.MUTED),
          ("70-80", F.C.AD), ("0-100", F.C.ACCENT)]

for name, colour in STRATA:
    d = bins[bins.age_bin == name].sort_values("drop_pct")
    n = int(d["n"].iloc[0])
    lbl = "full cohort" if name == "0-100" else f"{name} y"
    ax.plot(d["drop_pct"] * 100, d["auc"], color=colour, lw=1.3,
            marker="o", ms=3.4,
            ls="-" if name != "0-100" else "--",
            label=f"{lbl} (n = {n})")

ax.set_xlabel("Features removed (%)")
ax.set_ylabel("AUC")
ax.set_title("Within age strata")
ax.set_xticks([0, 10, 20, 30, 40])
ax.set_ylim(0.46, 0.92)
ax.legend(loc="lower right", handlelength=1.2, labelspacing=0.25,
          fontsize=5.8, frameon=True, facecolor="white", edgecolor="none",
          framealpha=0.95, borderpad=0.3)
F.panel_label(ax, "c")

print("[figure2_removal]")
F.save(fig, "figure2_removal")

print(f"    baseline {base.auc_point:.4f} -> peak {peak.auc_point:.4f} "
      f"at {peak.drop_pct:.0%}  ({peak.stars}, p = {peak.p_wald:.5f})")
print(f"    observed dAUC {b.observed:+.4f} vs null 95% "
      f"[{b.null_lo:+.4f}, {b.null_hi:+.4f}]  {b.p_formatted}")
peaks = []
for name, _ in STRATA:
    d = bins[bins.age_bin == name]
    peaks.append(f"{name}:{d.loc[d['auc'].idxmax(), 'drop_pct']:.0%}")
print(f"    peak threshold per stratum: {', '.join(peaks)}")
