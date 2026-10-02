"""Figure 1 — age-sensitivity profile and its effect on AD classification."""
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
sup = pd.read_csv(RES / "fig1_suppression.csv")
bee = pd.read_csv(RES / "age_model_shap_values.csv")

fig = plt.figure(figsize=(F.COL2, 3.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.18, 1.0, 1.0], wspace=0.85)

ax = fig.add_subplot(gs[0, 0])
order = (bee.groupby("feature")["rank"].first().sort_values().index.tolist())
order = order[::-1]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "bt", [F.C.CTRL, F.C.AD])
rng = np.random.default_rng(0)

for i, feat in enumerate(order):
    d = bee[bee.feature == feat]
    jitter = rng.uniform(-0.28, 0.28, len(d))
    ax.scatter(d["shap"], i + jitter, c=d["feature_value_norm"], cmap=cmap,
               s=1.4, alpha=0.55, linewidths=0, rasterized=True)

ax.axvline(0, color=F.C.ACCENT, lw=0.6, alpha=0.5)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order, fontsize=5.2)
ax.set_ylim(-0.8, len(order) + 1.4)
ax.set_xlabel("SHAP value (years)")
ax.set_title("Age-sensitivity profile")
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
cax.text(0.5, 2.1, "feature value", transform=cax.transAxes, fontsize=5.4,
         ha="center", va="bottom", color=F.C.ACCENT)
F.panel_label(ax, "a")

ax = fig.add_subplot(gs[0, 1])
k = sup["k"].values

ax.fill_between(k, sup["null_lo"], sup["null_hi"], color=F.C.CTRL,
                alpha=0.55, lw=0, label="random subsets, 95%")
ax.plot(k, sup["null_mean"], color=F.C.MUTED, lw=1.1, ls="--",
        label="random subsets, mean")
ax.plot(k, sup["age_auc"], color=F.C.AD, lw=1.4, marker="o", ms=3.2,
        label="age-ranked subset")
ax.axhline(0.5, color=F.C.ACCENT, lw=0.6, ls=":", alpha=0.7)
ax.text(3, 0.503, "chance", fontsize=5.8, ha="left", va="bottom",
        color=F.C.ACCENT)

ax.set_xlabel("Number of features")
ax.set_ylabel("AUC")
ax.set_title("AUC by subset size")
ax.set_xlim(0, 116)
ax.set_ylim(0.49, 0.74)
ax.legend(loc="upper right", handlelength=1.3, borderpad=0.3, fontsize=6.3,
          labelspacing=0.3, frameon=True, facecolor="white",
          edgecolor="none", framealpha=0.95)
F.panel_label(ax, "b")

ax = fig.add_subplot(gs[0, 2])

ax.fill_between(k, sup["null_lo"] - sup["null_mean"],
                sup["null_hi"] - sup["null_mean"],
                color=F.C.CTRL, alpha=0.55, lw=0,
                label="null 95%")
ax.axhline(0, color=F.C.ACCENT, lw=0.7)

sig = sup["p"] < cfg.ALPHA
ax.plot(k, sup["delta"], color=F.C.AD, lw=1.4, zorder=3)
ax.scatter(k[~sig.values], sup["delta"][~sig], s=18, color=F.C.AD,
           zorder=4, label="n.s.")
ax.scatter(k[sig.values], sup["delta"][sig], s=34, color=F.C.HILITE,
           marker="D", zorder=5, edgecolors="white", linewidths=0.5,
           label=f"p < {cfg.ALPHA:g}")

for _, r in sup[sig].iterrows():
    ax.annotate(f"k={int(r.k)}\np = {r.p:.4f}",
                xy=(r.k, r.delta), xytext=(r.k + 4, r.delta + 0.021),
                fontsize=5.8, color=F.C.HILITE, ha="left",
                arrowprops=dict(arrowstyle="-", lw=0.5, color=F.C.HILITE))

for kk, dx, dy in ((5, 8, 0.014), (11, 14, 0.004)):
    r = sup[sup.k == kk].iloc[0]
    ax.annotate(f"k={kk}, p = {r.p:.2f}",
                xy=(r.k, r.delta), xytext=(r.k + dx, r.delta + dy),
                fontsize=5.8, color=F.C.ACCENT, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.5, color=F.C.MUTED))

ax.set_xlabel("Number of features")
ax.set_ylabel("$\\Delta$AUC (age-ranked $-$ null mean)")
ax.set_title("Difference from random selection")
ax.set_xlim(0, 116)
ax.set_ylim(-0.098, 0.088)
ax.legend(loc="lower right", handlelength=1.2, borderpad=0.3,
          labelspacing=0.3, frameon=True, facecolor="white",
          edgecolor="none", framealpha=0.95)
F.panel_label(ax, "c")

print("[figure1_suppression]")
F.save(fig, "figure1_suppression")

s5 = sup[sup.k == 5].iloc[0]
s11 = sup[sup.k == 11].iloc[0]
s44 = sup[sup.k == 44].iloc[0]
print(f"    k=5   dAUC {s5.delta:+.4f}  p = {s5.p:.4f}")
print(f"    k=11  dAUC {s11.delta:+.4f}  p = {s11.p:.4f}")
print(f"    k=44  dAUC {s44.delta:+.4f}  p = {s44.p:.4f}  <- the only "
      f"significant point, and it is positive")
