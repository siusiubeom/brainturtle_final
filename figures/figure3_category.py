"""Figure 3 — which acoustic domains the age filter prunes."""
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
cat = pd.read_csv(RES / "fig3_category_rates.csv")
sub = pd.read_csv(RES / "fig3_mfcc_subfamily.csv")

global_rate = cat["dropped"].sum() / cat["total"].sum()

fig = plt.figure(figsize=(F.COL2, 2.8))
gs = fig.add_gridspec(1, 2, width_ratios=[1.7, 1.0], wspace=0.55)

ax = fig.add_subplot(gs[0, 0])
cat = cat.sort_values("rate", ascending=False).reset_index(drop=True)
colours = [F.C.AD if r > global_rate else F.C.CTRL for r in cat["rate"]]

bars = ax.bar(range(len(cat)), cat["rate"], color=colours,
              edgecolor=F.C.ACCENT, linewidth=0.6, width=0.68)

for i, r in cat.iterrows():
    ax.text(i, r["rate"] + 0.018,
            f"{int(r['dropped'])}/{int(r['total'])}",
            ha="center", va="bottom", fontsize=6.4, color=F.C.ACCENT)

ax.axhline(global_rate, color=F.C.HILITE, lw=1.0, ls="--", zorder=3)
ax.text(len(cat) - 0.42, global_rate + 0.012,
        f"overall {global_rate:.3f}  ({int(cat['dropped'].sum())}/"
        f"{int(cat['total'].sum())})",
        fontsize=6.2, color=F.C.HILITE, ha="right", va="bottom")

ax.set_xticks(range(len(cat)))
ax.set_xticklabels(cat["category"], fontsize=7)
ax.set_ylabel("Proportion of family removed")
ax.set_title("Preferential pruning by acoustic domain")
ax.set_ylim(0, 0.60)
ax.grid(axis="x", visible=False)
F.panel_label(ax, "a")

handles = [plt.Rectangle((0, 0), 1, 1, facecolor=F.C.AD,
                         edgecolor=F.C.ACCENT, lw=0.6),
           plt.Rectangle((0, 0), 1, 1, facecolor=F.C.CTRL,
                         edgecolor=F.C.ACCENT, lw=0.6)]
ax.legend(handles, ["above overall rate", "below overall rate"],
          loc="upper right", handlelength=1.1, borderpad=0.3)

ax = fig.add_subplot(gs[0, 1])
sub = sub.sort_values("rate", ascending=False).reset_index(drop=True)
LBL = {"shape": "shape\n(skew, kurtosis)", "non-shape": "non-shape\n(mean, SD, percentiles)"}

colours = [F.C.AD if r > global_rate else F.C.CTRL for r in sub["rate"]]
ax.bar(range(len(sub)), sub["rate"], color=colours,
       edgecolor=F.C.ACCENT, linewidth=0.6, width=0.55)

for i, r in sub.iterrows():
    ax.text(i, r["rate"] - 0.022,
            f"{int(r['dropped'])}/{int(r['total'])}\n{r['rate']:.1%}",
            ha="center", va="top", fontsize=6.4, color="white",
            fontweight="bold")

ax.axhline(global_rate, color=F.C.HILITE, lw=1.0, ls="--", zorder=3)
ax.text(len(sub) - 0.55, global_rate + 0.014, f"overall {global_rate:.3f}",
        fontsize=6.2, color=F.C.HILITE, ha="right", va="bottom")

ax.set_xticks(range(len(sub)))
ax.set_xticklabels([LBL[s] for s in sub["subfamily"]], fontsize=6.2)
ax.set_ylabel("Proportion removed")
ax.set_title("Within MFCC")
ax.set_ylim(0, 0.78)
ax.grid(axis="x", visible=False)
F.panel_label(ax, "b")

print("[figure3_category]")
F.save(fig, "figure3_category")

print(f"    overall removal rate {global_rate:.4f} "
      f"({int(cat['dropped'].sum())}/{int(cat['total'].sum())})")
for _, r in cat.iterrows():
    flag = "above" if r["rate"] > global_rate else "below"
    print(f"    {r['category']:14s} {int(r['dropped']):2d}/{int(r['total']):2d} "
          f"= {r['rate']:.3f}  ({flag}, enrichment {r['enrichment']:.2f})")
