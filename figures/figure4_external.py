"""Figure 5 — cross-lingual transfer, against the Kang age benchmark."""
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
ext = pd.read_csv(RES / "fig4_external.csv")
kang = pd.read_csv(RES / "kang_speaker_table.csv")
kbench = pd.read_csv(RES / "kang_age_benchmark.csv")


def bench(quantity):
    row = kbench[kbench["quantity"] == quantity]
    if not len(row):
        raise KeyError(quantity)
    return row.iloc[0]


AGE_AUC = bench("AUC, age alone")

fig = plt.figure(figsize=(F.COL1, 3.0))
gs = fig.add_gridspec(1, 1)

ax = fig.add_subplot(gs[0, 0])
CONDITIONS = [
    ("Kang", F.C.AD, "-", "full corpus"),
    ("Kang (investigator-removed)", F.C.MUTED, "--",
     "investigator turns removed"),
]

for cohort, colour, ls, label in CONDITIONS:
    d = sweep[sweep.cohort == cohort].sort_values("drop_pct")
    x = d["drop_pct"] * 100
    ax.fill_between(x, d["ci_low"], d["ci_high"], color=colour,
                    alpha=0.13, lw=0)
    ax.plot(x, d["auc"], color=colour, lw=1.4, ls=ls, zorder=3, label=label)

    above = d["above_chance"].values
    ax.scatter(x[above], d["auc"][above], s=26, color=colour,
               zorder=4, edgecolors="white", linewidths=0.6)
    ax.scatter(x[~above], d["auc"][~above], s=22, facecolor="white",
               edgecolors=colour, linewidths=0.9, zorder=4)

    peak = d.loc[d["auc"].idxmax()]
    ax.annotate(f"peak {peak.auc:.3f}\np = {peak.p:.4f}",
                xy=(peak.drop_pct * 100, peak.auc),
                xytext=(43.5, peak.auc + (0.045 if cohort == "Kang"
                                          else -0.045)),
                fontsize=6.2, color=colour, ha="left", va="center",
                fontweight="bold",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=colour,
                                shrinkA=0, shrinkB=2))

ax.axhline(0.5, color=F.C.ACCENT, lw=0.7, ls=":", zorder=2)
ax.text(0.6, 0.507, "chance", fontsize=6, color=F.C.ACCENT, va="bottom")

ax.axhspan(AGE_AUC.lo, AGE_AUC.hi, color=F.C.HILITE, alpha=0.10, lw=0,
           zorder=1)
ax.axhline(AGE_AUC.value, color=F.C.HILITE, lw=0.9, ls="-.", zorder=2)
ax.text(0.6, AGE_AUC.value + 0.008,
        f"age alone, {AGE_AUC.value:.3f}", fontsize=6,
        color=F.C.HILITE, va="bottom", fontweight="bold")

ax.set_xlabel("Age-sensitive features removed (%)")
ax.set_ylabel("AUC")
ax.set_title("Cross-lingual transfer (Kang, Korean)")
ax.set_xticks([0, 10, 20, 30, 40])
ax.set_xlim(-2, 62)
ax.set_ylim(0.30, 0.98)

marker_key = [
    plt.Line2D([], [], marker="o", ls="none", color=F.C.ACCENT, ms=4.5,
               label="above chance (p < 0.05)"),
    plt.Line2D([], [], marker="o", ls="none", markerfacecolor="white",
               markeredgecolor=F.C.ACCENT, ms=4.5, label="not above chance"),
]
leg1 = ax.legend(loc="upper left", handlelength=1.6, labelspacing=0.3,
                 frameon=True, facecolor="white", edgecolor="none",
                 framealpha=0.95)
ax.add_artist(leg1)
ax.legend(handles=marker_key, loc="lower right", handlelength=1.0,
          labelspacing=0.3, frameon=True, facecolor="white",
          edgecolor="none", framealpha=0.95)

print("[figure4_external]")
F.save(fig, "figure4_external")

for _, r in ext[ext.cohort.str.startswith("Kang")].iterrows():
    print(f"    {r['cohort']:30s} n={int(r.n_speakers):3d}  "
          f"AUC {r.auc:.4f} [{r.ci_low:.3f}, {r.ci_high:.3f}]  "
          f"{r.p_formatted}")
print(f"    {'age alone (Kang)':30s} n={len(kang):3d}  "
      f"AUC {AGE_AUC.value:.4f} [{AGE_AUC.lo:.3f}, {AGE_AUC.hi:.3f}]")
for q in ("AUC difference, 0% removal (undeconfounded) vs age",
          "AUC difference, 30% removal vs age",
          "AUC difference, 35% removal vs age"):
    r = bench(q)
    print(f"    {q:60s} {r.value:+.4f} [{r.lo:+.3f}, {r.hi:+.3f}]  "
          f"p = {r.p:.4f}")
for cohort, _, _, _ in CONDITIONS:
    d = sweep[sweep.cohort == cohort]
    pk = d.loc[d["auc"].idxmax()]
    first = d[d["above_chance"]]["drop_pct"].min()
    print(f"    {cohort:30s} peak {pk.auc:.4f} at {pk.drop_pct:.0%} "
          f"(p = {pk.p:.4f}); first above chance at {first:.0%}")
