"""Figure 1 — chronological age is the benchmark, not chance. Twice."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

import figstyle as F
from brainturtle import config as cfg

F.apply()

RES = cfg.RESULTS / f"n{cfg.N_PERM}"
bench = pd.read_csv(RES / "fig1_age_benchmark.csv")
roc = pd.read_csv(RES / "fig1_age_roc.csv")
spk = pd.read_csv(RES / "fig1_age_distribution.csv")
kbench = pd.read_csv(RES / "kang_age_benchmark.csv")
kspk = pd.read_csv(RES / "kang_speaker_table.csv")


def row(table, prefix):
    m = table[table["quantity"].str.startswith(prefix)]
    if m.empty:
        raise SystemExit(f"missing row: {prefix!r}")
    return m.iloc[0]


MODELS = [
    ("Age alone", F.C.HILITE, "--"),
    ("0% removal", F.C.MUTED, "-"),
    ("30% removal", F.C.AD, "-"),
]

fig = plt.figure(figsize=(F.COL2, 5.6))
gs = fig.add_gridspec(2, 3, wspace=0.85, hspace=0.80)


def panel_ages(ax, ages, labels, names, bins, gap, title, letter):
    for name, lab, colour in names:
        v = ages[labels == lab].dropna()
        ax.hist(v, bins=bins, alpha=0.75, color=colour,
                label=f"{name} (n = {len(v)})", edgecolor="white",
                linewidth=0.5)

    ymax = ax.get_ylim()[1]
    lo = ages[labels == names[0][1]].mean()
    hi = ages[labels == names[1][1]].mean()
    for _, lab, colour in names:
        ax.vlines(ages[labels == lab].mean(), 0, ymax * 1.08, color=colour,
                  lw=1.4, ls="--", zorder=3)
    ax.annotate("", xy=(hi, ymax * 1.08), xytext=(lo, ymax * 1.08),
                arrowprops=dict(arrowstyle="<->", color=F.C.ACCENT, lw=0.9))
    ax.text((lo + hi) / 2, ymax * 1.10, f"{abs(gap):.2f} y", ha="center",
            va="bottom", fontsize=7, fontweight="bold", color=F.C.ACCENT)
    ax.set_xlabel("Age at entry (years)")
    ax.set_ylabel("Speakers")
    ax.set_title(title)
    ax.set_ylim(0, ymax * 1.55)
    ax.legend(loc="upper right", fontsize=6.5, frameon=True,
              facecolor="white", edgecolor="none", framealpha=0.9,
              borderpad=0.3)
    F.panel_label(ax, letter)


panel_ages(fig.add_subplot(gs[0, 0]), spk["entryage"], spk["label"],
           [("Control", 0, F.C.CTRL), ("AD", 1, F.C.AD)],
           np.arange(45, 95, 5), row(bench, "age gap")["value"],
           "Pitt: age structure", "a")

panel_ages(fig.add_subplot(gs[1, 0]), kspk["age"], kspk["label"],
           [("HC", 0, F.C.CTRL), ("MCI", 1, F.C.AD)],
           np.arange(58, 84, 2), row(kbench, "age gap")["value"],
           "Kang: age structure", "d")


def panel_roc(ax, curves, title, letter):
    ax.plot([0, 1], [0, 1], color=F.C.GRID, lw=1.0, ls=":", zorder=1)
    for label, colour, ls, fpr, tpr, auc in curves:
        ax.plot(fpr, tpr, color=colour, lw=1.5, ls=ls, zorder=3, label=label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=6.5, handlelength=1.4,
              borderpad=0.3, labelspacing=0.35)
    F.panel_label(ax, letter)


pitt_curves = []
for label, colour, ls in MODELS:
    d = roc[roc["model"] == label]
    if d.empty:
        raise SystemExit(f"missing ROC series: {label!r}")
    pitt_curves.append((label, colour, ls, d["fpr"].values, d["tpr"].values,
                        d["auc"].iloc[0]))
panel_roc(fig.add_subplot(gs[0, 1]), pitt_curves, "Pitt: held-out ROC", "b")

ky = kspk["label"].to_numpy().astype(int)
kang_curves = []
for (label, colour, ls), col, key in zip(
        MODELS, ("age", "drop0", "drop30"),
        ("AUC, age alone", "AUC, 0% removal", "AUC, 30% removal")):
    fpr, tpr, _ = roc_curve(ky, kspk[col].to_numpy())
    kang_curves.append((label, colour, ls, fpr, tpr, row(kbench, key)["value"]))
panel_roc(fig.add_subplot(gs[1, 1]), kang_curves, "Kang: external ROC", "e")


def panel_auc(ax, table, keys, diffs, title, letter):
    ys = np.arange(len(keys))[::-1]
    XLO, XHI, XEND = 0.30, 1.0, 1.15
    for yi, ((label, colour, _), key) in zip(ys, zip(MODELS, keys)):
        r = row(table, key)
        ax.plot([r["lo"], r["hi"]], [yi, yi], color=colour, lw=1.4, zorder=3)
        for x in (r["lo"], r["hi"]):
            ax.plot([x, x], [yi - 0.12, yi + 0.12], color=colour, lw=1.4,
                    zorder=3)
        ax.plot(r["value"], yi, marker="o", ms=6.5, color=colour,
                markeredgecolor="white", markeredgewidth=0.8, zorder=4)
        ax.text(XEND, yi, f"{r['value']:.3f}", va="center", ha="right",
                fontsize=6.5, fontweight="bold", color=F.C.ACCENT)

    ax.vlines(0.5, -0.5, len(MODELS) - 0.4, color=F.C.GRID, lw=1.0, ls=":",
              zorder=1)
    ax.vlines(row(table, keys[0])["value"], -0.5, len(MODELS) - 0.4,
              color=F.C.HILITE, lw=0.9, ls="-.", zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([m[0] for m in MODELS])
    ax.set_xlim(XLO, XEND)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0])
    ax.spines["bottom"].set_bounds(XLO, XHI)
    ax.set_xlabel("AUC (95% CI)")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)

    rows = []
    for caption, key in diffs:
        r = row(table, key)
        rows.append((caption, f"{r['value']:+.4f}",
                     f"({r['lo']:+.4f}, {r['hi']:+.4f})",
                     f"p = {r['p']:.3g}"))
    ax.set_ylim(-1.7, len(MODELS) - 0.4)
    F.text_table(ax, XLO + 0.015, -0.8, rows, fontsize=6.5)
    F.panel_label(ax, letter)


panel_auc(fig.add_subplot(gs[0, 2]), bench,
          ["AUC, age alone", "AUC, 0% removal", "AUC, 30% removal"],
          [("0% vs age", "AUC difference, 0% model vs age"),
           ("30% vs age", "AUC difference, 30% model vs age")],
          "Pitt: AUC, 95% CI", "c")

panel_auc(fig.add_subplot(gs[1, 2]), kbench,
          ["AUC, age alone", "AUC, 0% removal", "AUC, 30% removal"],
          [("0% vs age", "AUC difference, 0% removal (undeconfounded) vs age"),
           ("30% vs age", "AUC difference, 30% removal vs age")],
          "Kang: AUC, 95% CI", "f")

F.save(fig, "figure1_age_benchmark")

print("[figure1_age_benchmark]")
for name, table in (("Pitt", bench), ("Kang", kbench)):
    a = row(table, "AUC, age alone")
    print(f"    {name:5s} age alone {a['value']:.4f} "
          f"[{a['lo']:.3f}, {a['hi']:.3f}]   gap "
          f"{row(table, 'age gap')['value']:+.2f} y")
