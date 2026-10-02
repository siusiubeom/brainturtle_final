"""Shared figure style and multi-format export."""
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from brainturtle import config as _cfg

FIGDIR = _cfg.FIGURES
FORMATS = {
    "tiff": dict(dpi=600, pil_kwargs={"compression": "tiff_lzw"}),
    "png": dict(dpi=300),
    "jpg": dict(dpi=300, pil_kwargs={"quality": 95}),
}


class C:
    """Palette, unchanged from pub_theme so old and new panels match."""
    AD = "#2C5F8A"
    CTRL = "#A8C4DC"
    ACCENT = "#1A3F5C"
    GRID = "#D8E4ED"
    HILITE = "#C1666B"
    MUTED = "#8A9BA8"
    OK = "#5B8C5A"


PALETTE = {0: C.CTRL, 1: C.AD}

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica Neue", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "axes.titlepad": 8.0,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.edgecolor": C.ACCENT,
    "axes.labelcolor": C.ACCENT,
    "text.color": C.ACCENT,
    "xtick.color": C.ACCENT,
    "ytick.color": C.ACCENT,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "axes.grid": True,
    "grid.color": C.GRID,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.7,
    "legend.frameon": False,
    "legend.fontsize": 7,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "lines.linewidth": 1.2,
    "patch.linewidth": 0.8,
}

COL1, COL2 = 3.50, 7.20

TITLE_PAD = RC["axes.titlepad"]
SUBTITLE_PAD = 17.0
LETTER_SIZE = 10
LETTER_GAP = 5.0

_LABELS = []


def apply():
    mpl.rcParams.update(RC)


def panel_label(ax, letter, **_):
    """Bold lowercase panel letter, npj house style; placed at save()."""
    if not letter.endswith(")"):
        letter += ")"
    _LABELS.append((ax, letter))


def _column(ax):
    ss = ax.get_subplotspec()
    if ss is None:
        return id(ax)
    return (id(ss.get_gridspec()), ss.colspan.start)


def _place_labels(fig):
    todo = [(ax, s) for ax, s in _LABELS if ax.figure is fig]
    if not todo:
        return
    for ax, _ in todo:
        ax.title.set_x(0.0)
        ax.title.set_ha("left")
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pt = fig.dpi / 72.0
    W, H = fig.bbox.width, fig.bbox.height
    for ax, s in todo:
        t = ax.title
        x0, yb = t.get_transform().transform(t.get_position())
        txt = fig.text(0, 0, s, fontsize=LETTER_SIZE, fontweight="bold",
                       color=C.ACCENT, ha="left", va="baseline")
        w = txt.get_window_extent(r).width
        txt.set_position(((x0 - LETTER_GAP * pt - w) / W, yb / H))


def text_table(ax, x, y, rows, fontsize=6.5, color=None, colgap=5.0,
               linespacing=1.45, transform=None, **kw):
    """Left-aligned text columns; rows is a list of tuples of strings."""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    tr = transform if transform is not None else ax.transData
    x0, y0 = tr.transform((x, y))
    pt = fig.dpi / 72.0
    lh = fontsize * linespacing * pt
    W, H = fig.bbox.width, fig.bbox.height
    cx = x0
    out = []
    for c in range(max(len(rw) for rw in rows)):
        wmax = 0.0
        for i, rw in enumerate(rows):
            if c >= len(rw):
                continue
            t = fig.text(cx / W, (y0 - i * lh) / H, rw[c], fontsize=fontsize,
                         color=color or C.ACCENT, ha="left", va="top", **kw)
            wmax = max(wmax, t.get_window_extent(r).width)
            out.append(t)
        cx += wmax + colgap * pt
    return out


def stars(p):
    """Significance markers. 'ns' rather than a bare gap, so it is explicit."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def save(fig, name, directory=None, close=True):
    """Write TIFF + PNG + JPG and print each path."""
    _place_labels(fig)
    d = Path(directory) if directory else FIGDIR
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext, kw in FORMATS.items():
        path = d / f"{name}.{ext}"
        fig.savefig(path, format=("tiff" if ext == "tiff" else ext), **kw)
        size_mb = os.path.getsize(path) / 1e6
        print(f"    -> {path.name}  ({size_mb:.2f} MB)")
        paths.append(path)
    if close:
        plt.close(fig)
    return paths
