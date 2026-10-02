"""Result persistence and consistent console output."""

import pandas as pd

from . import config as cfg


def outdir(n_perm=None):
    d = cfg.RESULTS / f"n{n_perm or cfg.N_PERM}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(df, name, n_perm=None):
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    path = outdir(n_perm) / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"  -> {path}")
    return path


def header(title, subtitle=None):
    print()
    print("=" * 78)
    print(title)
    if subtitle:
        print(subtitle)
    print("=" * 78)


def run_banner(script):
    print(f"[{script}]  N_PERM={cfg.N_PERM}  N_BOOT={cfg.N_BOOT}  "
          f"SEEDS={cfg.SEEDS}  stability={cfg.N_PERM_SEEDS} seeds "
          f"x {cfg.N_PERM_STABILITY}")


def show(df, cols=None, fmt="{:.4f}"):
    print(df.to_string(index=False, columns=cols,
                       float_format=lambda v: fmt.format(v)))


def stability_note(rows, label="p"):
    """One line summarising whether a p moved across permutation seeds."""
    ps = [r["p"] for r in rows]
    zs = [r["z"] for r in rows]
    print(f"  stability over {len(rows)} permutation seeds: "
          f"{label} {min(ps):.4g}-{max(ps):.4g}, z {min(zs):+.2f}..{max(zs):+.2f}")
