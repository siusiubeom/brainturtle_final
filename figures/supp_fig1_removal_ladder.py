"""Supplementary Figure S1 — fine-grained age-feature removal sweep."""
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
lad = pd.read_csv(RES / "fig2a_removal_ladder.csv").sort_values("drop_pct")
N_TOTAL = int(lad["n_features_kept"].iloc[0])

x = lad["drop_pct"].values * 100.0
auc = lad["auc_point"].values
lo = lad["ci_low"].values
hi = lad["ci_high"].values
pre = lad["in_prespecified"].values.astype(bool)

peak_i = int(np.argmax(auc))
peak_x, peak_y = x[peak_i], auc[peak_i]

fig, ax = plt.subplots(figsize=(F.COL1 * 1.2, 3.3))

ax.fill_between(x, lo, hi, color=F.C.CTRL, alpha=0.55, lw=0,
                label="bootstrap 95% CI", zorder=1)

base = auc[0]
ax.axhline(base, color=F.C.MUTED, lw=0.8, ls=":", zorder=2)
ax.text(12.5, base - 0.003, "0% baseline", fontsize=5.8, ha="center",
        va="top", color=F.C.MUTED)

ax.axvline(peak_x, color=F.C.HILITE, lw=0.8, ls="--", alpha=0.75, zorder=2)

ax.plot(x, auc, color=F.C.AD, lw=1.5, zorder=4, label="AUC")
ax.scatter(x[pre], auc[pre], s=26, color=F.C.AD, zorder=5,
           edgecolors="white", linewidths=0.6, label="pre-specified grid")
ax.scatter(x[~pre], auc[~pre], s=22, facecolors="white",
           edgecolors=F.C.AD, linewidths=1.0, zorder=5,
           label="exploratory only")
ax.scatter([peak_x], [peak_y], s=58, marker="D", color=F.C.HILITE,
           edgecolors="white", linewidths=0.7, zorder=6, label="peak")

for xi, yi, st in zip(x, auc, lad["stars"].values):
    if st in ("-", "ns") or pd.isna(st):
        continue
    dy = 0.016 if xi != peak_x else 0.021
    ax.text(xi, yi + dy, str(st), fontsize=8, fontweight="bold",
            ha="center", va="bottom", color=F.C.HILITE, zorder=7)

ax.annotate(f"peak {peak_y:.3f}\n"
            f"{int(lad['n_features_kept'].iloc[peak_i])} of {N_TOTAL} features kept",
            xy=(peak_x - 0.7, peak_y + 0.004),
            xytext=(0.5, 0.815), fontsize=6.2, color=F.C.HILITE,
            ha="left", va="top",
            arrowprops=dict(arrowstyle="-", lw=0.6, color=F.C.HILITE))

ax.set_xlabel("Age-sensitive features removed (% of SHAP ranking)")
ax.set_ylabel("AUC (Pitt, speaker-held-out)")
ax.set_title("Fine-grained removal sweep")
ax.set_xticks(x)
ax.set_xticklabels([f"{v:.0f}" for v in x])
ax.set_xlim(-2.5, 42.5)
ax.set_ylim(0.578, 0.818)

leg = ax.legend(loc="lower center", ncol=2, handlelength=1.4, borderpad=0.3,
                labelspacing=0.28, columnspacing=1.1, frameon=True,
                facecolor="white", edgecolor="none", framealpha=0.95)
leg.set_zorder(8)


print("[supplementary_fig1_removal_ladder]")
F.save(fig, "supplementary_fig1_removal_ladder")

print(f"    peak AUC {peak_y:.4f} at {peak_x:.0f}% removal "
      f"({int(lad['n_features_kept'].iloc[peak_i])} of {N_TOTAL} features kept)")
for _, r in lad.iterrows():
    print(f"    {r.drop_pct*100:5.0f}%  n={int(r.n_features_kept):3d}  "
          f"AUC {r.auc_point:.4f}  CI [{r.ci_low:.4f}, {r.ci_high:.4f}]  "
          f"{r.stars}")
