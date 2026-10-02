"""Supplementary Figure 14 — is the low-feature-count effect specific to"""
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

CANDIDATES = sorted(
    (int(p.parent.name[1:]), p)
    for p in cfg.RESULTS.glob("n*/item8_detector_robustness.csv")
    if p.parent.name[1:].isdigit()
)
if not CANDIDATES:
    raise SystemExit("no item8_detector_robustness.csv found; run "
                     "analyses/k_detector_robustness.py")
N_PERM_SWEEP, SRC = CANDIDATES[-1]
if N_PERM_SWEEP < cfg.N_PERM:
    print(f"  note: sweep is at {N_PERM_SWEEP:,} permutations, below the "
          f"paper's {cfg.N_PERM:,}")

d = pd.read_csv(SRC)
DETECTORS = ["IsolationForest", "LOF", "Mahalanobis", "PCA-reconstruction"]
NICE = {"IsolationForest": "Isolation Forest", "LOF": "Local Outlier Factor",
        "Mahalanobis": "Mahalanobis distance",
        "PCA-reconstruction": "PCA reconstruction error"}
RANKINGS = list(d["ranking"].unique())
KS = sorted(d["k"].unique())

fig = plt.figure(figsize=(F.COL2, 3.3))
gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.35, 1.0], wspace=0.65)

COLOURS = {"IsolationForest": F.C.AD, "LOF": F.C.HILITE,
           "Mahalanobis": F.C.OK, "PCA-reconstruction": F.C.MUTED}
MARKERS = {"IsolationForest": "o", "LOF": "s",
           "Mahalanobis": "^", "PCA-reconstruction": "D"}

for col, ranking in enumerate(RANKINGS):
    ax = fig.add_subplot(gs[0, col])
    sub = d[d.ranking == ranking]
    ax.axhline(0, color=F.C.ACCENT, lw=0.8, zorder=2)
    ax.axhspan(ax.get_ylim()[0], 0, color=F.C.CTRL, alpha=0.22, lw=0,
               zorder=1)

    for det in DETECTORS:
        s = sub[sub.detector == det].sort_values("k")
        ax.plot(s["k"], s["delta"], color=COLOURS[det], lw=1.1,
                marker=MARKERS[det], ms=3.8, label=NICE[det].replace("\n", " "),
                zorder=3)
        sig = s[s["p"] < cfg.ALPHA]
        ax.scatter(sig["k"], sig["delta"], s=58, facecolor="none",
                   edgecolor=COLOURS[det], lw=1.2, zorder=4)

    ax.set_xlabel("Number of features")
    if col == 0:
        ax.set_ylabel("$\\Delta$AUC (age-ranked $-$ null mean)")
    ax.set_title(f"{ranking} ranking")
    ax.set_xticks(KS)
    ax.set_ylim(-0.115, 0.082)
    ax.text(0.03, 0.04, "age-ranked worse than random",
            transform=ax.transAxes, fontsize=5.6, color=F.C.ACCENT,
            style="italic")
    if col == 0:
        ax.legend(loc="upper left", handlelength=1.4, labelspacing=0.25,
                  fontsize=5.8, frameon=True, facecolor="white",
                  edgecolor="none", framealpha=0.95)
    F.panel_label(ax, "ab"[col])

ax = fig.add_subplot(gs[0, 2])
agree = (d.groupby(["ranking", "k"])["worse_than_random"].sum()
         .reset_index().rename(columns={"worse_than_random": "n_below"}))

width = 0.38
xs = np.arange(len(KS))
for i, ranking in enumerate(RANKINGS):
    s = agree[agree.ranking == ranking].set_index("k").reindex(KS)
    colour = F.C.AD if i == 0 else F.C.CTRL
    bars = ax.bar(xs + (i - 0.5) * width, s["n_below"], width=width,
                  color=colour, edgecolor=F.C.ACCENT, lw=0.6,
                  label=ranking)
    for x, v in zip(xs + (i - 0.5) * width, s["n_below"]):
        ax.text(x, v + 0.08, f"{int(v)}", ha="center", va="bottom",
                fontsize=6, color=F.C.ACCENT)

ax.axhline(4, color=F.C.HILITE, lw=0.9, ls="--", zorder=3)

ax.set_xticks(xs)
ax.set_xticklabels(KS)
ax.set_xlabel("Number of features")
ax.set_ylabel("Detectors placing age-ranked\nbelow the random null")
ax.set_title("Sign agreement")
ax.set_ylim(0, 5.0)
ax.set_yticks([0, 1, 2, 3, 4])
ax.grid(axis="x", visible=False)
ax.legend(loc="upper right", handlelength=1.1, fontsize=5.8,
          frameon=True, facecolor="white", edgecolor="none", framealpha=0.95)
F.panel_label(ax, "c")

fig.text(0.5, -0.035,
         f"Open circles mark p < {cfg.ALPHA:g}. "
         f"{N_PERM_SWEEP:,} permutations per detector, ranking and feature "
         f"count.",
         ha="center", fontsize=6, color=F.C.ACCENT)

print("[supp_fig14_detector_robustness]")
F.save(fig, "supplementary_fig14_detector_robustness")

for ranking in RANKINGS:
    s = agree[agree.ranking == ranking].set_index("k")
    print(f"    {ranking:18s} detectors below null: "
          + ", ".join(f"k={k}:{int(s.loc[k, 'n_below'])}/4" for k in KS))
sig = d[d["p"] < cfg.ALPHA]
print(f"    {len(sig)} of {len(d)} cells reach p < {cfg.ALPHA:g}; "
      f"{int((sig['delta'] < 0).sum())} of those are suppression, "
      f"{int((sig['delta'] > 0).sum())} enhancement")
