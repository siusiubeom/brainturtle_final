"""Supplementary Figure 10 — calibration of the 0% and 30% removal models."""
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
bins = pd.read_csv(RES / "calibration_bins.csv")
cal = pd.read_csv(RES / "calibration.csv").set_index("model")
conv = pd.read_csv(RES / "ece_conventions.csv")

MODELS = ["0% removal", "30% removal"]
COLOR = {"0% removal": F.C.MUTED, "30% removal": F.C.AD}

PINNED = ("equal-width", 10)

LEFT, RIGHT, TOP, BOTTOM, WSPACE, RATIOS = 0.07, 0.985, 0.88, 0.16, 0.60, (1.0, 1.28)
_w = F.COL2 * (RIGHT - LEFT) / (sum(RATIOS) + WSPACE * sum(RATIOS) / 2)
fig = plt.figure(figsize=(F.COL2, _w / (TOP - BOTTOM)))
gs = fig.add_gridspec(1, 2, width_ratios=list(RATIOS), wspace=WSPACE,
                      left=LEFT, right=RIGHT, top=TOP, bottom=BOTTOM)

ax = fig.add_subplot(gs[0, 0])
ax.plot([0, 1], [0, 1], ls="--", lw=0.9, color=F.C.ACCENT, alpha=0.6,
        zorder=1, label="perfect calibration")

for m in MODELS:
    d = bins[bins.model == m]
    ax.plot(d["mean_predicted"], d["observed_frequency"], color=COLOR[m],
            lw=1.1, alpha=0.85, zorder=2)
    ax.scatter(d["mean_predicted"], d["observed_frequency"],
               s=d["n"] * 1.15, color=COLOR[m], alpha=0.85,
               edgecolors="white", linewidths=0.6, zorder=3, label=m)

ax.set_xlim(-0.03, 1.03)
ax.set_ylim(-0.03, 1.03)
ax.set_aspect("equal")
ax.set_xticks(np.arange(0, 1.01, 0.25))
ax.set_yticks(np.arange(0, 1.01, 0.25))
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed frequency of AD")
ax.set_title("Reliability")

F.text_table(ax, 0.035, 0.975,
             [(f"{m}:", f"Brier {cal.loc[m, 'brier']:.4f}",
               f"ECE {cal.loc[m, 'ece']:.4f}") for m in MODELS],
             fontsize=5.9, transform=ax.transAxes)

leg = ax.legend(loc="lower right", handlelength=1.2, borderpad=0.35,
                labelspacing=0.35, scatterpoints=1, frameon=True,
                facecolor="white", edgecolor="none", framealpha=0.95)
for h in leg.legend_handles:
    if hasattr(h, "set_sizes"):
        h.set_sizes([22])
ax.text(0.045, 0.805, "marker area $\\propto$ speakers in bin",
        transform=ax.transAxes, fontsize=5.4, ha="left", va="top",
        color=F.C.MUTED)
F.panel_label(ax, "a")

ax = fig.add_subplot(gs[0, 1])
conv = conv.sort_values(["scheme", "n_bins"], ascending=[False, True])
conv = conv.reset_index(drop=True)
xpos = np.arange(len(conv), dtype=float)
w = 0.38

ax.bar(xpos - w / 2, conv["ece_0%"], width=w, color=F.C.MUTED, alpha=0.9,
       edgecolor="white", label="0% removal", zorder=3)
ax.bar(xpos + w / 2, conv["ece_30%"], width=w, color=F.C.AD, alpha=0.95,
       edgecolor="white", label="30% removal", zorder=3)

pin = int(conv.index[(conv["scheme"] == PINNED[0])
                     & (conv["n_bins"] == PINNED[1])][0])
ax.axvspan(pin - 0.5, pin + 0.5, ymax=0.82, color=F.C.GRID, alpha=0.55,
           lw=0, zorder=1)
ax.annotate("pinned convention", xy=(pin, 0.1735), xytext=(pin, 0.190),
            fontsize=5.8, ha="center", va="bottom", color=F.C.ACCENT,
            zorder=5,
            arrowprops=dict(arrowstyle="-|>,head_width=0.12,head_length=0.28",
                            lw=0.6, color=F.C.ACCENT))

ax.set_xticks(xpos)
ax.set_xticklabels(conv["n_bins"].astype(int))
ax.set_xlim(-0.7, len(conv) - 0.3)
ax.set_ylim(0, 0.252)
ax.set_xlabel("Number of bins")
ax.set_ylabel("Expected calibration error")
ax.set_title("ECE depends on the binning convention")
ax.grid(axis="x", visible=False)

n_per = len(conv) // 2
ax.vlines(n_per - 0.5, 0, 0.252 * 0.82, color=F.C.GRID, lw=0.8, zorder=2)
for i, name in enumerate(conv["scheme"].unique()):
    centre = (i * n_per) + (n_per - 1) / 2
    ax.text(centre, -0.155, name, transform=ax.get_xaxis_transform(),
            fontsize=6.4, ha="center", va="top", color=F.C.ACCENT)

ax.set_yticks(np.arange(0, 0.201, 0.05))

bar_handles = [h for h in ax.containers]
ax.legend(handles=list(bar_handles),
          loc="upper left", ncol=2, handlelength=1.4, borderpad=0.3,
          labelspacing=0.3, columnspacing=1.2, fontsize=6.2, frameon=True,
          facecolor="white", edgecolor="none", framealpha=0.95)
F.panel_label(ax, "b")

print("[supplementary_fig5_calibration]")
F.save(fig, "supplementary_fig5_calibration")

for m in MODELS:
    print(f"    {m:12s}  AUC {cal.loc[m, 'auc']:.4f}  "
          f"Brier {cal.loc[m, 'brier']:.4f}  ECE {cal.loc[m, 'ece']:.4f}")
print(f"    ECE 0%  across conventions: "
      f"{conv['ece_0%'].min():.4f} - {conv['ece_0%'].max():.4f}")
print(f"    ECE 30% across conventions: "
      f"{conv['ece_30%'].min():.4f} - {conv['ece_30%'].max():.4f}")
