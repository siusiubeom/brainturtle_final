"""Supplementary Figure 11 — recording duration by diagnostic label."""
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
rec = pd.read_csv(RES / "recording_stats.csv")
summ = pd.read_csv(RES / "recording_stats_summary.csv").set_index("metric")
row = summ.loc["duration_sec"]

METRIC = "duration_sec"
GROUPS = [(0, "Control", F.C.CTRL), (1, "AD", F.C.AD)]

vals = [rec.loc[rec["label"] == lab, METRIC].values for lab, _, _ in GROUPS]
pos = [0, 1]

rng = np.random.default_rng(cfg.BOOT_SEED)

fig, ax = plt.subplots(figsize=(F.COL1, 3.5))

vp = ax.violinplot(vals, positions=pos, widths=0.72, showextrema=False,
                   showmedians=False)
for body, (_, _, colour) in zip(vp["bodies"], GROUPS):
    body.set_facecolor(colour)
    body.set_alpha(0.30)
    body.set_edgecolor(colour)
    body.set_linewidth(0.8)
    body.set_zorder(2)

for p, v, (_, _, colour) in zip(pos, vals, GROUPS):
    jitter = rng.uniform(-0.115, 0.115, size=v.size)
    ax.scatter(p + jitter, v, s=4.5, color=colour, alpha=0.30,
               linewidths=0, zorder=3)

bp = ax.boxplot(vals, positions=pos, widths=0.18, showfliers=False,
                patch_artist=True, zorder=4,
                medianprops=dict(color=F.C.HILITE, lw=1.6),
                boxprops=dict(facecolor="white", edgecolor=F.C.ACCENT, lw=0.8),
                whiskerprops=dict(color=F.C.ACCENT, lw=0.8),
                capprops=dict(color=F.C.ACCENT, lw=0.8))

meds = [float(row["median_control"]), float(row["median_ad"])]
sds = [float(row["sd_control"]), float(row["sd_ad"])]
ns = [int(row["n_control"]), int(row["n_ad"])]
outs = [int(row["upper_outliers_control"]), int(row["upper_outliers_ad"])]
p_loc = float(row["p_location"])
p_lev = float(row["levene_p_spread"])
rb = float(row["rank_biserial"])

ax.set_ylim(0, 335)

for p, m, sd, n in zip(pos, meds, sds, ns):
    ax.text(p + 0.155, m, f"{m:.1f} s", fontsize=6.6, fontweight="bold",
            ha="left", va="center", color=F.C.HILITE, zorder=6,
            bbox=dict(boxstyle="square,pad=0.15", facecolor="white",
                      edgecolor="none", alpha=0.85))

label_x = [-0.46, 1.60]
label_ha = ["center", "right"]
for p, o, v, lx, ha in zip(pos, outs, vals, label_x, label_ha):
    ax.annotate(f"{o} upper\noutliers", xy=(p, v.max()),
                xytext=(lx, v.max() - 4), fontsize=6.0, ha=ha,
                va="top", color=F.C.MUTED,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=F.C.MUTED,
                                shrinkB=2))

y = 296
ax.plot([0, 0, 1, 1], [y, y + 8, y + 8, y], lw=0.8, color=F.C.ACCENT,
        zorder=6, clip_on=False)
ax.text(0.5, y + 11, f"{F.stars(p_loc)}  Mann-Whitney p = {p_loc:.1e}",
        fontsize=6.4, ha="center", va="bottom", color=F.C.HILITE,
        fontweight="bold")

ax.set_xticks(pos)
ax.set_xticklabels([f"{name}\n(n = {n})" for (_, name, _), n in zip(GROUPS, ns)])
ax.set_xlim(-0.62, 1.62)
ax.set_ylabel("Recording duration (s)")
ax.set_title("Recording duration by diagnostic label")
ax.grid(axis="x", visible=False)


print("[supplementary_fig11_recording_duration]")
F.save(fig, "supplementary_fig11_recording_duration")

print(f"    median  control {meds[0]:.2f} s  ->  AD {meds[1]:.2f} s")
print(f"    SD      control {sds[0]:.2f} s  ->  AD {sds[1]:.2f} s   "
      f"Levene p = {p_lev:.5f}")
print(f"    upper outliers  control {outs[0]}  ->  AD {outs[1]}")
print(f"    Mann-Whitney U = {row['mannwhitney_u']:.0f}, p = {p_loc:.3e}, "
      f"rank-biserial {rb:+.3f}")
