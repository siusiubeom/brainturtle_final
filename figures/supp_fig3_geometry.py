"""Supplementary Figure 9 — geometry of the acoustic feature space."""
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
emb = pd.read_csv(RES / "geometry_embedding.csv")
sil = pd.read_csv(RES / "geometry_silhouettes.csv")
summ = pd.read_csv(RES / "geometry_summary.csv")

DECON = next(s for s in summ["space"] if s.startswith("deconfounded"))
FULL = next(s for s in summ["space"] if s.startswith("full"))
N_DECON = int(summ.loc[summ["space"] == DECON, "n_features"].iloc[0])
LABELS = {0: ("Control", F.C.CTRL), 1: ("AD", F.C.AD)}

hd = summ.set_index("space")
assert set(hd.index) == {DECON, FULL}, hd.index.tolist()
for sp in (DECON, FULL):
    assert hd.loc[sp, "hdbscan_clusters"] == 0, (sp, hd.loc[sp])
    assert hd.loc[sp, "hdbscan_noise"] == hd.loc[sp, "n_speakers"], (sp, hd.loc[sp])
n_noise = int(hd.loc[DECON, "hdbscan_noise"])
n_spk = int(hd.loc[DECON, "n_speakers"])

d = emb[emb.space == DECON]

print("[supp_fig3_geometry]")
print(f"    deconfounded space: {int(hd.loc[DECON,'n_features'])} features, "
      f"{n_spk} speakers, PCA dim {int(hd.loc[DECON,'pca_dim'])}")
print(f"    HDBSCAN: 0 clusters, {n_noise}/{n_spk} noise in BOTH spaces")

fig = plt.figure(figsize=(F.COL2, 3.05))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.16], wspace=0.72,
                      left=0.06, right=0.985, top=0.86, bottom=0.155)

ax = fig.add_subplot(gs[0, 0])
for lab, (name, col) in LABELS.items():
    s = d[d.label == lab]
    ax.scatter(s.pc1, s.pc2, s=11, color=col, alpha=0.8, linewidths=0.3,
               edgecolors="white", label=f"{name} (n = {len(s)})",
               rasterized=True)
ax.set_ylim(d.pc2.min() - 1.5, d.pc2.max() + 9.0)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("PCA", pad=F.SUBTITLE_PAD)
ax.text(0.0, 1.015, f"deconfounded space, {N_DECON} features", transform=ax.transAxes,
        ha="left", va="bottom", fontsize=6, style="italic", color=F.C.MUTED)
ax.legend(loc="upper right", handlelength=0.9, borderpad=0.3,
          labelspacing=0.28, handletextpad=0.4, frameon=True,
          facecolor="white", edgecolor="none", framealpha=0.9)
F.panel_label(ax, "a")

ax = fig.add_subplot(gs[0, 1])
for lab, (name, col) in LABELS.items():
    s = d[d.label == lab]
    ax.scatter(s.umap1, s.umap2, s=11, color=col, alpha=0.8, linewidths=0.3,
               edgecolors="white", label=name, rasterized=True)
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.set_title("UMAP", pad=F.SUBTITLE_PAD)
ax.text(0.0, 1.015, f"deconfounded space, {N_DECON} features", transform=ax.transAxes,
        ha="left", va="bottom", fontsize=6, style="italic", color=F.C.MUTED)
F.panel_label(ax, "b")

ax = fig.add_subplot(gs[0, 2])
style = {DECON: (F.C.AD, "o", "-"), FULL: (F.C.MUTED, "s", "--")}
means = {}
for sp in (DECON, FULL):
    s = sil[sil.space == sp].sort_values("k")
    col, mk, ls = style[sp]
    m = s.silhouette.mean()
    means[sp] = m
    ax.plot(s.k, s.silhouette, color=col, marker=mk, ms=3.4, lw=1.3, ls=ls,
            label=f"{sp}\nmean over K = {m:.3f}", zorder=3)
    ax.axhline(m, color=col, lw=0.7, ls=":", alpha=0.75, zorder=1)

ax.axhline(0.5, color=F.C.HILITE, lw=0.8, ls="--", alpha=0.85, zorder=2)
ax.text(1.85, 0.512, "0.5 = conventional floor for real structure",
        fontsize=5.6, color=F.C.HILITE, ha="left", va="bottom")

ax.set_xlabel("K (k-means clusters)")
ax.set_ylabel("Silhouette score")
ax.set_title("No cluster structure in either space", pad=F.SUBTITLE_PAD)
ax.set_xticks([2, 3, 4, 5, 6])
ax.set_xlim(1.75, 6.25)
ax.set_ylim(0.0, 0.74)
ax.legend(loc="upper right", bbox_to_anchor=(1.005, 1.0), handlelength=1.4,
          borderpad=0.3, labelspacing=0.45, handletextpad=0.5, fontsize=5.9,
          frameon=True, facecolor="white", edgecolor="none", framealpha=0.92)

ax.text(0.04, 0.60,
        f"HDBSCAN: 0 clusters in both spaces\n"
        f"(all {n_noise}/{n_spk} speakers labelled noise)",
        transform=ax.transAxes, ha="left", va="top", fontsize=6.0,
        color=F.C.ACCENT,
        bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                  edgecolor=F.C.GRID, lw=0.6))
F.panel_label(ax, "c")

for sp in (DECON, FULL):
    print(f"    {sp:28s} mean silhouette K=2..6 = {means[sp]:.4f}  "
          f"(K=2 {sil[(sil.space == sp) & (sil.k == 2)].silhouette.iloc[0]:.3f})")

F.save(fig, "supplementary_fig3_geometry")
