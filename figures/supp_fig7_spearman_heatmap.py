"""Supplementary Figure S7 — age sensitivity of all 110 acoustic features."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import figstyle as F
from brainturtle import config as cfg
from brainturtle import data, featuresets as fs

F.apply()

RHO_SCREEN = cfg.LEGACY_UNION_RHO

cv = data.commonvoice()
age = cv["age_num_x"].values
feats = fs.common_features()

rows = []
for f in feats:
    r = spearmanr(cv[f].values, age, nan_policy="omit")
    rho, p = float(r.statistic), float(r.pvalue)
    rows.append({"feature": f, "rho": rho, "p": p,
                 "family": fs.categorize(f)})

d = pd.DataFrame(rows)
d["abs_rho"] = d["rho"].abs()
d = d.sort_values("abs_rho", ascending=True).reset_index(drop=True)
n_over = int((d["abs_rho"] >= RHO_SCREEN).sum())

POS, NEG = F.C.AD, F.C.HILITE
colors = [POS if r > 0 else (NEG if r < 0 else F.C.MUTED) for r in d["rho"]]

fig, ax = plt.subplots(figsize=(F.COL1, 8.6))
y = np.arange(len(d))
ax.barh(y, d["abs_rho"], height=0.72, color=colors, edgecolor="none",
        zorder=3)

ax.axvline(RHO_SCREEN, color=F.C.ACCENT, lw=0.9, ls="--", alpha=0.85,
           zorder=4)
ax.axvspan(RHO_SCREEN, 0.235, color=F.C.GRID, alpha=0.45, lw=0, zorder=1)

ax.set_yticks(y)
ax.set_yticklabels(d["feature"], fontsize=4.0)
for tick, a in zip(ax.get_yticklabels(), d["abs_rho"]):
    if a >= RHO_SCREEN:
        tick.set_fontweight("bold")
    else:
        tick.set_color(F.C.MUTED)
ax.tick_params(axis="y", length=1.5, pad=1.5)
ax.set_ylim(-0.9, len(d) - 0.1)
ax.set_xlim(0, 0.235)
ax.set_xticks(np.arange(0, 0.226, 0.05))
ax.set_xlabel("|Spearman $\\rho$| with age (Common Voice, n = 417)")
ax.set_title(f"Age sensitivity of the {len(d)} shared features")
ax.grid(axis="y", visible=False)

ax.text(RHO_SCREEN + 0.008, len(d) - n_over - 2.0,
        f"$|\\rho| \\geq$ {RHO_SCREEN:.2f}\n{n_over} of {len(d)} features",
        fontsize=6.4, ha="left", va="top", color=F.C.ACCENT,
        linespacing=1.4, zorder=5)

handles = [plt.Rectangle((0, 0), 1, 1, color=POS),
           plt.Rectangle((0, 0), 1, 1, color=NEG)]
ax.legend(handles, ["$\\rho > 0$ (rises with age)",
                    "$\\rho < 0$ (falls with age)"],
          loc="lower right", bbox_to_anchor=(1.0, 0.008), handlelength=1.0,
          handleheight=0.9, borderpad=0.35, labelspacing=0.35, fontsize=6.0,
          frameon=True, facecolor="white", edgecolor="none", framealpha=0.95)

ax.text(0.5, -0.072,
        "Correlations are against a tied 7-level ordinal (decade midpoints), not continuous age.",
        transform=ax.transAxes, fontsize=5.4, ha="center", va="top",
        color=F.C.MUTED, linespacing=1.5)

print("[supplementary_fig7_spearman_heatmap]")
F.save(fig, "supplementary_fig7_spearman_heatmap")

top = d.sort_values("abs_rho", ascending=False).head(10)
print(f"    n = {len(cv)} Common Voice speakers; age levels "
      f"{sorted(cv['age_num_x'].dropna().unique().astype(int).tolist())}")
print(f"    {n_over} of {len(d)} features with |rho| >= {RHO_SCREEN:.2f}")
print(f"    max |rho| = {d['abs_rho'].max():.4f} "
      f"({top.iloc[0]['feature']})")
print("    top 10 by |rho|:")
for _, r in top.iterrows():
    print(f"      {r.feature:>16s}  rho {r.rho:+.4f}  p {r.p:.2e}")
