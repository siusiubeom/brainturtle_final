"""Supplementary Figure S8 — DementiaBank Pitt cohort demographics."""
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
from brainturtle import data

F.apply()

p = data.pitt_with_age()
cv = data.commonvoice()

ad = p.loc[p["label"] == 1, "entryage"].dropna().values
ct = p.loc[p["label"] == 0, "entryage"].dropna().values
n_ad, n_ct = int((p["label"] == 1).sum()), int((p["label"] == 0).sum())

pitt_lo, pitt_hi = float(p["entryage"].min()), float(p["entryage"].max())
cv_age = cv["age_num_x"].dropna()
in_range = int(((cv_age >= pitt_lo) & (cv_age <= pitt_hi)).sum())
pct_in_range = 100.0 * in_range / len(cv_age)
top_band = float(cv_age.max())
n_top = int((cv_age == top_band).sum())
pct_top = 100.0 * n_top / len(cv_age)

fig = plt.figure(figsize=(F.COL2, 2.95))
gs = fig.add_gridspec(1, 3, width_ratios=[1.30, 0.72, 1.22], wspace=0.75)

ax = fig.add_subplot(gs[0, 0])
edges = np.arange(45, 91, 5.0)
w = (edges[1] - edges[0]) / 2 * 0.86
h_ad, _ = np.histogram(ad, bins=edges)
h_ct, _ = np.histogram(ct, bins=edges)
centres = edges[:-1] + (edges[1] - edges[0]) / 2

ax.bar(centres - w / 2, h_ct, width=w, color=F.C.CTRL, edgecolor="white",
       label=f"Control (n = {n_ct})", zorder=3)
ax.bar(centres + w / 2, h_ad, width=w, color=F.C.AD, edgecolor="white",
       label=f"AD (n = {n_ad})", zorder=3)

ymax = max(h_ad.max(), h_ct.max())
for v, c in ((ct.mean(), F.C.MUTED), (ad.mean(), F.C.HILITE)):
    ax.vlines(v, 0, ymax * 1.02, color=c, lw=1.0, ls="--", zorder=4)

ax.set_ylim(0, ymax * 1.66)
ax.annotate(f"control mean\n{ct.mean():.1f} y", xy=(ct.mean(), ymax * 1.02),
            xytext=(ct.mean() - 7.8, ymax * 1.21), fontsize=5.8,
            color=F.C.MUTED, ha="center", va="top", linespacing=1.3,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=F.C.MUTED))
ax.annotate(f"AD mean\n{ad.mean():.1f} y", xy=(ad.mean(), ymax * 1.02),
            xytext=(ad.mean() + 8.2, ymax * 1.21), fontsize=5.8,
            color=F.C.HILITE, ha="center", va="top", linespacing=1.3,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=F.C.HILITE))

ax.set_xlabel("Age at entry (years)")
ax.text(0.0, 1.015, f"all {n_ad + n_ct} speaker records",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=6.0, style="italic", color=F.C.MUTED)
ax.set_ylabel("Speakers")
ax.set_title("Age by diagnosis", pad=F.SUBTITLE_PAD)
ax.set_xticks(np.arange(45, 91, 10))
ax.set_xlim(43, 92)
ax.grid(axis="x", visible=False)
ax.legend(loc="upper left", handlelength=1.0, handleheight=0.9,
          borderpad=0.3, labelspacing=0.3, frameon=True, facecolor="white",
          edgecolor="none", framealpha=0.95)
F.panel_label(ax, "a")

ax = fig.add_subplot(gs[0, 1])
counts = [n_ct, n_ad]
ax.bar([0, 1], counts, width=0.62, color=[F.C.CTRL, F.C.AD],
       edgecolor="white", zorder=3)
total = n_ct + n_ad
for i, v in enumerate(counts):
    ax.text(i, v + total * 0.018, f"{v}\n{100 * v / total:.1f}%",
            ha="center", va="bottom", fontsize=6.4, color=F.C.ACCENT,
            linespacing=1.3)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Control", "AD"])
ax.set_xlim(-0.62, 1.62)
ax.set_ylim(0, max(counts) * 1.30)
ax.set_ylabel("Speakers")
ax.set_title("Class balance", pad=F.SUBTITLE_PAD)
ax.grid(axis="x", visible=False)
ax.text(0.5, -0.155, f"n = {total} speaker records",
        transform=ax.transAxes, fontsize=5.8, ha="center", va="top",
        color=F.C.MUTED)
F.panel_label(ax, "b")

ax = fig.add_subplot(gs[0, 2])
bands = cv_age.value_counts().sort_index()
ax.axvspan(pitt_lo, pitt_hi, color=F.C.GRID, alpha=0.75, lw=0, zorder=1)
cols = [F.C.AD if (b >= pitt_lo and b <= pitt_hi) else F.C.MUTED
        for b in bands.index]
ax.bar(bands.index, bands.values, width=7.4, color=cols, alpha=0.92,
       edgecolor="white", zorder=3)
for b, v in bands.items():
    ax.text(b, v + len(cv_age) * 0.012, str(int(v)), ha="center",
            va="bottom", fontsize=5.8, color=F.C.ACCENT, zorder=4)

ax.set_ylim(0, bands.max() * 1.55)
ax.set_xlim(8, 95)
ax.set_xticks([int(b) for b in bands.index])
ax.set_xlabel("Common Voice age band (decade midpoint)")
ax.set_ylabel("Speakers")
ax.set_title("Normative reference vs clinical range", pad=F.SUBTITLE_PAD)
ax.grid(axis="x", visible=False)

ax.text(80.0, bands.max() * 0.95,
        f"Pitt age range\n{pitt_lo:.0f}-{pitt_hi:.0f} y", fontsize=5.8,
        ha="center", va="top", color=F.C.ACCENT, linespacing=1.3, zorder=5)
F.panel_label(ax, "c")

print("[supplementary_fig8_pitt_demographics]")
F.save(fig, "supplementary_fig8_pitt_demographics")

print(f"    speaker records {len(p)}  unique speaker_id "
      f"{p['speaker_id'].nunique()}")
print(f"    AD      n = {n_ad:3d}  age {ad.mean():.2f} +/- {ad.std(ddof=1):.2f} "
      f"(range {ad.min():.0f}-{ad.max():.0f})")
print(f"    Control n = {n_ct:3d}  age {ct.mean():.2f} +/- {ct.std(ddof=1):.2f} "
      f"(range {ct.min():.0f}-{ct.max():.0f})")
print(f"    mean age gap AD - control = {ad.mean() - ct.mean():+.2f} years")
print(f"    Common Voice: {len(cv_age)} speakers, bands "
      f"{[int(b) for b in bands.index]}")
print(f"    within Pitt range [{pitt_lo:.0f}, {pitt_hi:.0f}]: {in_range} "
      f"({pct_in_range:.1f}%);  oldest band ({top_band:.0f}): {n_top} "
      f"({pct_top:.1f}%)")
