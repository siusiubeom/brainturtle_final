"""Supplementary Figure 2 — stability of the age-sensitivity ranking."""
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
from brainturtle import config as cfg, featuresets as fs

F.apply()

RES = cfg.RESULTS / f"n{cfg.N_PERM}"
bee = pd.read_csv(RES / "age_model_shap_values.csv")
chk = pd.read_csv(RES / "age_model_ranking_check.csv")
prt = pd.read_csv(RES / "age_model_partition.csv")

N_FEATURES = len(fs.common_features())

VAL = "validation fit (Fig 1a)"
FULLFIT = "full-corpus fit (S13)"
chk = chk.set_index("fit")
part = prt.iloc[0]

print("[supp_fig13_age_ranking_stability]")
print(f"    validation fit  vs stored ranking : rho = "
      f"{chk.loc[VAL, 'spearman_rank_rho']:.3f}  "
      f"{int(chk.loc[VAL, 'overlap_top33'])}/33 at the cut")
print(f"    full-corpus fit vs stored ranking : rho = "
      f"{chk.loc[FULLFIT, 'spearman_rank_rho']:.3f}  "
      f"{int(chk.loc[FULLFIT, 'overlap_top33'])}/33 at the cut")
print(f"    validation fit  vs full-corpus fit: rho = "
      f"{part.spearman_rank_rho:.3f}  "
      f"{int(part.overlap_top33)}/33 at the cut   <- the S13 claim")

fig = plt.figure(figsize=(F.COL2, 3.55))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.28], wspace=0.60,
                      left=0.115, right=0.985, top=0.865, bottom=0.145)

ax = fig.add_subplot(gs[0, 0])
order = bee.groupby("feature")["rank"].first().sort_values().index.tolist()
order = order[::-1]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "bt", [F.C.CTRL, F.C.AD])
rng = np.random.default_rng(0)

for i, feat in enumerate(order):
    s = bee[bee.feature == feat]
    jitter = rng.uniform(-0.28, 0.28, len(s))
    ax.scatter(s["shap"], i + jitter, c=s["feature_value_norm"], cmap=cmap,
               vmin=0, vmax=1, s=1.4, alpha=0.55, linewidths=0,
               rasterized=True)

ax.axvline(0, color=F.C.ACCENT, lw=0.6, alpha=0.5)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order, fontsize=5.2)
ax.set_ylim(-0.85, len(order) + 1.4)
ax.set_xlabel("SHAP value (years)")
ax.set_title("Age-model SHAP profile")
ax.grid(axis="y", visible=False)

cax = ax.inset_axes([0.60, 0.925, 0.38, 0.020])
sm = plt.cm.ScalarMappable(cmap=cmap,
                           norm=matplotlib.colors.Normalize(0, 1))
cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
cb.set_ticks([])
cb.outline.set_visible(False)
cax.text(-0.06, 0.5, "low", transform=cax.transAxes, fontsize=5.4,
         ha="right", va="center", color=F.C.ACCENT)
cax.text(1.06, 0.5, "high", transform=cax.transAxes, fontsize=5.4,
         ha="left", va="center", color=F.C.ACCENT)
cax.text(0.5, 2.2, "feature value", transform=cax.transAxes, fontsize=5.4,
         ha="center", va="bottom", color=F.C.ACCENT)
F.panel_label(ax, "a")

ax = fig.add_subplot(gs[0, 1])

KS = [5, 11, 20, 33]
rows = [
    ("validation fit\nvs full-corpus fit",
     part.spearman_rank_rho,
     {k: part.get(f"overlap_top{k}", np.nan) for k in KS},
     F.C.HILITE, True),
    ("validation fit\nvs stored ranking",
     chk.loc[VAL, "spearman_rank_rho"],
     {k: chk.loc[VAL].get(f"overlap_top{k}", np.nan) for k in KS},
     F.C.MUTED, False),
    ("full-corpus fit\nvs stored ranking",
     chk.loc[FULLFIT, "spearman_rank_rho"],
     {k: chk.loc[FULLFIT].get(f"overlap_top{k}", np.nan) for k in KS},
     F.C.AD, False),
]

nrow = len(rows)
bar_h = 0.17
offs = np.linspace(-1.5, 1.5, len(KS)) * bar_h
shades = [0.42, 0.60, 0.78, 1.0]

for r, (name, rho, ov, col, key) in enumerate(rows):
    y0 = nrow - 1 - r
    for j, k in enumerate(KS):
        v = ov[k]
        if not np.isfinite(v):
            continue
        frac = v / k
        ax.barh(y0 + offs[j], frac, height=bar_h * 0.86, color=col,
                alpha=shades[j], lw=0, zorder=3)
        ax.text(frac + 0.012, y0 + offs[j], f"{int(v)}/{k}", fontsize=5.6,
                va="center", ha="left", color=F.C.ACCENT, zorder=4)
    ax.text(-0.035, y0, name, fontsize=6.4, va="center", ha="right",
            color=col if key else F.C.ACCENT,
            fontweight="bold" if key else "normal")
    ax.text(1.30, y0, f"$\\rho$ = {rho:.2f}", fontsize=10 if key else 7.5,
            fontweight="bold", va="center", ha="center",
            color=col if key else F.C.ACCENT)

handles = [plt.Rectangle((0, 0), 1, 1, color=F.C.MUTED, alpha=shades[j])
           for j in range(len(KS))]
ax.legend(handles, [f"top-{k}" for k in KS], loc="upper left",
          bbox_to_anchor=(-0.005, 1.005), ncol=4, handlelength=0.9,
          handleheight=0.7, columnspacing=0.9, handletextpad=0.4,
          borderpad=0.25, fontsize=6, frameon=False)

ax.axvline(1.0, color=F.C.OK, lw=0.8, ls="--", alpha=0.9, zorder=2)
ax.text(1.012, -0.74, "perfect agreement", fontsize=5.8,
        color=F.C.OK, ha="left", va="bottom")

ax.set_xlim(0, 1.42)
ax.set_ylim(-0.80, nrow + 0.88)
ax.set_yticks([])
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xticklabels(["0", "25", "50", "75", "100%"])
ax.set_xlabel("Shared features at the top-k cut")
ax.set_title("Measured agreement between the fits")
ax.spines["left"].set_visible(False)
ax.grid(axis="y", visible=False)
ax.tick_params(axis="y", length=0)
ax.text(1.30, nrow + 0.42, f"rank $\\rho$\n({N_FEATURES} features)", fontsize=6,
        fontweight="bold", ha="center", va="center", color=F.C.ACCENT)
F.panel_label(ax, "b")

F.save(fig, "supplementary_fig13_age_ranking_stability")
