"""Supplementary Figure 6 — gender-feature removal against a random null."""
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
draws = pd.read_csv(RES / "suppfig9_null_draws.csv")
s = pd.read_csv(RES / "suppfig9_summary.csv").iloc[0]

null = draws["null_diff"].values
obs = float(s["observed"])
lo, hi = float(s["null_lo"]), float(s["null_hi"])
n_ge, n_perm = int(s["n_ge_observed"]), int(s["n_perm"])
frac_ge = n_ge / n_perm

fig, ax = plt.subplots(figsize=(F.COL1 * 1.45, 3.35))

bins = np.linspace(null.min(), null.max(), 61)
counts, edges, patches = ax.hist(null, bins=bins, color=F.C.CTRL,
                                 edgecolor="white", linewidth=0.3, zorder=2)

ymax = counts.max()
ax.set_ylim(0, ymax * 1.85)
xlo, xhi = null.min() * 1.10, null.max() * 1.18
ax.set_xlim(xlo, xhi)

ax.axvspan(obs, xhi, color=F.C.MUTED, alpha=0.16, lw=0, zorder=1)

pad = (xhi - xlo) * 0.008
for v, lab, ha, dx in ((lo, "null 2.5%", "right", -pad),
                       (hi, "null 97.5%", "left", pad)):
    ax.vlines(v, 0, ymax * 1.12, color=F.C.ACCENT, lw=0.9, ls="--", zorder=4)
    ax.text(v + dx, ymax * 1.05, lab, fontsize=5.8, ha=ha, va="center",
            color=F.C.ACCENT, zorder=5,
            bbox=dict(boxstyle="square,pad=0.15", facecolor="white",
                      edgecolor="none", alpha=0.9))

ax.vlines(0.0, 0, ymax * 1.12, color=F.C.MUTED, lw=0.7, ls=":", zorder=3)

ax.vlines(obs, 0, ymax * 1.12, color=F.C.HILITE, lw=1.8, zorder=6)
ax.annotate("observed\ngender-targeted\n%+.4f" % obs,
            xy=(obs, ymax * 0.50), xytext=(xlo * 0.97, ymax * 0.80),
            fontsize=6.4, color=F.C.HILITE, ha="left", va="top",
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", lw=0.7, color=F.C.HILITE))

verdict = "\n".join([
    "observed lies INSIDE the null 95% interval",
    f"two-sided p = {s['p']:.4f}    (z = {s['z']:+.2f})",
    f"{n_ge:,} / {n_perm:,} random removals ({frac_ge:.1%})",
    "did as well or better",
    f"AUC {s['auc_baseline']:.4f} -> {s['auc_gender_targeted']:.4f}",
])
ax.text(0.99, 0.985, verdict, transform=ax.transAxes, fontsize=6.2,
        ha="right", va="top", color=F.C.ACCENT, linespacing=1.5, zorder=8,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor=F.C.MUTED, linewidth=0.5, alpha=0.95))

ax.set_xlabel(r"$\Delta$AUC after removing 30% of features (removed $-$ baseline)")
ax.set_ylabel(f"Random removals (of {n_perm:,})")
ax.set_title("Gender-feature removal is indistinguishable from chance")

handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor=F.C.CTRL, edgecolor="white",
                  linewidth=0.3),
    plt.Rectangle((0, 0), 1, 1, facecolor=F.C.MUTED, alpha=0.30,
                  edgecolor="none"),
    plt.Line2D([], [], color=F.C.HILITE, lw=1.8),
]
labels = [
    f"random 30% removal (null, {n_perm:,} draws)",
    f"random draws at or above observed ({frac_ge:.1%})",
    "observed, gender-SHAP-ranked removal",
]
leg = ax.legend(handles, labels, loc="upper left", handlelength=1.5,
                borderpad=0.35, labelspacing=0.32, frameon=True,
                facecolor="white", edgecolor="none", framealpha=0.95)
leg.set_zorder(9)

print("[supplementary_fig9_gender_control]")
F.save(fig, "supplementary_fig9_gender_control")

print(f"    observed dAUC {obs:+.4f}  null mean {s['null_mean']:+.4f} "
      f"sd {s['null_sd']:.4f}")
print(f"    null 95% [{lo:+.4f}, {hi:+.4f}]  observed inside: "
      f"{bool(s['observed_inside_null_95'])}")
print(f"    p = {s['p']:.4f}   {n_ge:,}/{n_perm:,} ({frac_ge:.1%}) "
      f"random removals did as well or better")
print("    published legend claims a meaningful decrease; there is none")
