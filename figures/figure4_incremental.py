"""Figure 4 — the improvement is not explained by age, in either corpus."""
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
pitt = pd.read_csv(RES / "fig4_incremental.csv")
kang = pd.read_csv(RES / "fig4_kang_incremental.csv")

DIFF = "difference of increments (deconfounded - undeconfounded)"

PITT_MODELS = [
    ("age alone", "Age alone", F.C.HILITE),
    ("age + undeconfounded score", "Age +\nundeconf.", F.C.MUTED),
    ("age + deconfounded score", "Age +\ndeconf.\n(30%)", F.C.AD),
]
KANG_MODELS = [
    ("age alone", "Age alone", F.C.HILITE),
    ("age + 0% removal (undeconfounded) score", "Age +\nundeconf.", F.C.MUTED),
    ("age + 30% removal score", "Age +\ndeconf.\n(30%)", F.C.CTRL),
    ("age + 35% removal score", "Age +\ndeconf.\n(35%)", F.C.AD),
]

PITT_INC = [
    ("age + undeconfounded score", "Undeconfounded", F.C.MUTED),
    ("age + deconfounded score", "Deconfounded", F.C.AD),
    (DIFF, "Difference", F.C.HILITE),
]
KANG_INC = [
    ("age + 0% removal (undeconfounded) score", "Undeconfounded", F.C.MUTED),
    ("age + 30% removal score", "Deconfounded (30%)", F.C.CTRL),
    ("age + 35% removal score", "Deconfounded (35%)", F.C.AD),
    (DIFF, "Difference", F.C.HILITE),
]


def row(table, key):
    m = table[table["model"] == key]
    if m.empty:
        raise SystemExit(f"missing row: {key!r}")
    return m.iloc[0]


def panel_absolute(ax, table, models, title, letter, ylim):
    """Absolute out-of-fold AUC, one bar per model, with 95% CIs."""
    xs = np.arange(len(models))
    for x, (key, label, colour) in zip(xs, models):
        r = row(table, key)
        ax.plot([x, x], [r["lo"], r["hi"]], color=colour, lw=1.4, zorder=3)
        for y in (r["lo"], r["hi"]):
            ax.plot([x - 0.12, x + 0.12], [y, y], color=colour, lw=1.4,
                    zorder=3)
        ax.plot(x, r["auc"], marker="o", ms=6.5, color=colour,
                markeredgecolor="white", markeredgewidth=0.8, zorder=4)
        ax.text(x, r["hi"] + 0.012, f"{r['auc']:.3f}", ha="center",
                va="bottom", fontsize=6.5, fontweight="bold", color=F.C.ACCENT)
    ax.axhline(0.5, color=F.C.GRID, lw=1.0, ls=":", zorder=1)
    ax.set_xlim(-0.6, len(models) - 0.4)
    ax.set_xticks(xs)
    ax.set_xticklabels([m[1] for m in models])
    ax.set_ylim(*ylim)
    ax.set_ylabel("Out-of-fold AUC (95% CI)")
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    F.panel_label(ax, letter)


def panel_increment(ax, table, items, title, letter):
    """Increment over age for each score, and the difference between two."""
    ys = np.arange(len(items))[::-1]
    for y, (key, label, colour) in zip(ys, items):
        r = row(table, key)
        ax.plot([r["inc_lo"], r["inc_hi"]], [y, y], color=colour, lw=1.4,
                zorder=3)
        for x in (r["inc_lo"], r["inc_hi"]):
            ax.plot([x, x], [y - 0.12, y + 0.12], color=colour, lw=1.4,
                    zorder=3)
        ax.plot(r["increment"], y, marker="o", ms=6.5, color=colour,
                markeredgecolor="white", markeredgewidth=0.8, zorder=4)
        txt = f"{r['increment']:+.4f}"
        if not np.isnan(r["p"]):
            txt += f"  {F.stars(r['p'])}"
        ax.text(r["inc_hi"] + 0.008, y, txt, va="center", ha="left",
                fontsize=6.5, fontweight="bold", color=F.C.ACCENT)

    ax.vlines(0, -0.5, len(items) - 0.4, color=F.C.ACCENT, lw=0.9, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([i[1] for i in items])
    ax.set_xlabel("Δ AUC over age alone (95% CI)")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)

    d = row(table, DIFF)
    ax.set_xlim(min(0.0, float(table["inc_lo"].min())) - 0.02,
                float(table["inc_hi"].max()) + 0.085)
    ax.set_ylim(-1.9, len(items) - 0.4)
    F.text_table(ax, 0.0, -0.9,
                 [("difference",
                   f"{d['increment']:+.4f}  ({d['inc_lo']:+.4f}, {d['inc_hi']:+.4f})"),
                  ("", f"p = {d['p']:.4f}")], fontsize=6.5)
    F.panel_label(ax, letter)


fig = plt.figure(figsize=(F.COL2, 5.6))
gs = fig.add_gridspec(2, 2, wspace=0.90, hspace=0.80, width_ratios=[1.0, 1.0])

panel_absolute(fig.add_subplot(gs[0, 0]), pitt, PITT_MODELS,
               "Pitt: models containing age", "a", (0.45, 0.97))
panel_increment(fig.add_subplot(gs[0, 1]), pitt, PITT_INC,
                "Pitt: increment over age", "b")
panel_absolute(fig.add_subplot(gs[1, 0]), kang, KANG_MODELS,
               "Kang: models containing age", "c", (0.45, 1.05))
panel_increment(fig.add_subplot(gs[1, 1]), kang, KANG_INC,
                "Kang: increment over age", "d")

F.save(fig, "figure4_incremental")

print("[figure4_incremental]")
for name, table in (("Pitt", pitt), ("Kang", kang)):
    d = row(table, DIFF)
    print(f"    {name:5s} difference of increments {d['increment']:+.4f} "
          f"[{d['inc_lo']:+.4f}, {d['inc_hi']:+.4f}]  p = {d['p']:.4f}")
