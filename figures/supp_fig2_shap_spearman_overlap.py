"""Supplementary Figure 8 — do SHAP and Spearman pick the same features?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from scipy.stats import spearmanr

import figstyle as F
from brainturtle import config as cfg
from brainturtle import data, featuresets as fs

F.apply()

RES = cfg.RESULTS / f"n{cfg.N_PERM}"
TOP_K = 30

feats = fs.common_features()
cv = data.commonvoice()
age = cv["age_num_x"].values

rho = {f: spearmanr(cv[f].values, age, nan_policy="omit").statistic
       for f in feats}
defined = [f for f in feats if np.isfinite(rho[f])]

sp_order = sorted(defined, key=lambda x: -abs(rho[x]))
shap = fs.load_shap_ranking()
shap = shap[shap.feature.isin(feats)].reset_index(drop=True)
shap_order = shap["feature"].tolist()

top_shap = set(shap_order[:TOP_K])
top_sp = set(sp_order[:TOP_K])
shared = [f for f in shap_order[:TOP_K] if f in top_sp]
n_shared = len(shared)
n_shap_only = TOP_K - n_shared
n_sp_only = TOP_K - n_shared

sp_rank = {f: i + 1 for i, f in enumerate(sp_order)}
shap_rank = {f: i + 1 for i, f in enumerate(shap_order)}
paired = [f for f in defined]
rank_rho = spearmanr([shap_rank[f] for f in paired],
                     [sp_rank[f] for f in paired])

print("[supp_fig2_shap_spearman_overlap]")
print(f"    features: {len(feats)}   speakers: {len(cv)}")
print(f"    top-{TOP_K} overlap: {n_shared} of {TOP_K} "
      f"({100 * n_shared / TOP_K:.0f}%)")
print(f"    ranking Spearman rho over {len(paired)} features: "
      f"{rank_rho.statistic:.3f}  (p = {rank_rho.pvalue:.2g})")

fig = plt.figure(figsize=(F.COL2, 4.05))
gs = fig.add_gridspec(1, 2, width_ratios=[1.02, 1.0], wspace=0.60,
                      left=0.055, right=0.985, top=0.905, bottom=0.155)

ax = fig.add_subplot(gs[0, 0])
ax.set_aspect("equal")
ax.set_anchor("N")
ax.set_xlim(-2.25, 2.25)
ax.set_ylim(-3.58, 1.02)
ax.axis("off")

R, CY, DX = 0.78, 0.05, 0.46
ax.add_patch(Circle((-DX, CY), R, facecolor=F.C.AD, alpha=0.28,
                    edgecolor=F.C.AD, lw=1.1, zorder=2))
ax.add_patch(Circle((DX, CY), R, facecolor=F.C.HILITE, alpha=0.24,
                    edgecolor=F.C.HILITE, lw=1.1, zorder=2))

ax.text(-1.05, CY, str(n_shap_only), ha="center", va="center",
        fontsize=13, fontweight="bold", color=F.C.AD, zorder=4)
ax.text(1.05, CY, str(n_sp_only), ha="center", va="center",
        fontsize=13, fontweight="bold", color=F.C.HILITE, zorder=4)
ax.text(0.0, CY + 0.14, str(n_shared), ha="center", va="center",
        fontsize=15, fontweight="bold", color=F.C.ACCENT, zorder=4)
ax.text(0.0, CY - 0.30, f"of {TOP_K}\n({100 * n_shared / TOP_K:.0f}%)",
        ha="center", va="center", fontsize=6.2, color=F.C.ACCENT, zorder=4)

ax.text(-DX - 0.42, CY - R - 0.13, f"top-{TOP_K}\nby |SHAP|", ha="center",
        va="top", fontsize=7.5, fontweight="bold", color=F.C.AD)
ax.text(DX + 0.42, CY - R - 0.13, f"top-{TOP_K}\nby |Spearman $\\rho$|",
        ha="center", va="top", fontsize=7.5, fontweight="bold",
        color=F.C.HILITE)

ax.text(0.0, -1.66, f"shared features ({n_shared})", ha="center", va="top",
        fontsize=6.8, fontweight="bold", color=F.C.ACCENT)
ncol, rowh = 3, 0.255
nrow = int(np.ceil(n_shared / ncol))
xs = [-1.50, 0.0, 1.50]
for i, feat in enumerate(shared):
    c, r = i // nrow, i % nrow
    ax.text(xs[c], -1.98 - r * rowh, feat, ha="center", va="top",
            fontsize=5.6, color=F.C.ACCENT, family="monospace")

ax.set_title("Overlap of the two age-sensitivity rankings")
F.panel_label(ax, "a")

ax = fig.add_subplot(gs[0, 1])

d = pd.DataFrame({"feature": defined})
d["absrho"] = d.feature.map(lambda f: abs(rho[f]))
d["shapval"] = d.feature.map(dict(zip(shap.feature, shap.mean_abs_shap)))
d["nsets"] = d.feature.map(lambda f: (f in top_shap) + (f in top_sp))

groups = [(2, "in both top-30", F.C.ACCENT, 26, "o"),
          (1, "in one top-30", F.C.AD, 16, "o"),
          (0, "in neither", F.C.MUTED, 11, "o")]
for val, lab, col, size, mk in groups:
    s = d[d.nsets == val]
    ax.scatter(s.absrho, s.shapval, s=size, marker=mk, color=col,
               alpha=0.85 if val else 0.55, linewidths=0.4,
               edgecolors="white" if val == 2 else "none",
               label=f"{lab} (n = {len(s)})", zorder=4 - val * -1)

ax.axvline(abs(rho[sp_order[TOP_K - 1]]), color=F.C.HILITE, lw=0.8, ls="--",
           alpha=0.8, zorder=1)
ax.axhline(d.shapval[d.feature.isin(top_shap)].min(), color=F.C.AD, lw=0.8,
           ls="--", alpha=0.8, zorder=1)
ax.text(abs(rho[sp_order[TOP_K - 1]]) + 0.003, 1.85, f"$\\rho$ top-{TOP_K} cut",
        fontsize=5.8, color=F.C.HILITE, ha="left", va="top", rotation=90)
ax.text(0.229, d.shapval[d.feature.isin(top_shap)].min() + 0.04,
        f"SHAP top-{TOP_K} cut", fontsize=5.8, color=F.C.AD, ha="right",
        va="bottom")

ax.set_xlabel("|Spearman $\\rho$| with age")
ax.set_ylabel("mean |SHAP| of the age model (years)")
ax.set_title("Where the two methods disagree")
ax.set_xlim(-0.006, 0.232)
ax.set_ylim(-0.08, 2.42)
ax.legend(loc="upper left", handlelength=1.0, borderpad=0.35,
          labelspacing=0.32, handletextpad=0.45, frameon=True,
          facecolor="white", edgecolor="none", framealpha=0.9)

ax.text(0.985, 0.035,
        f"ranking Spearman $\\rho$ = {rank_rho.statistic:.2f}\n"
        f"(all {len(paired)} features, p = {rank_rho.pvalue:.1e})",
        transform=ax.transAxes, fontsize=6.4, ha="right", va="bottom",
        color=F.C.ACCENT,
        bbox=dict(boxstyle="round,pad=0.34", facecolor="white",
                  edgecolor=F.C.GRID, lw=0.6))
F.panel_label(ax, "b")

F.save(fig, "supplementary_fig2_shap_spearman_overlap")
