"""Explicit panel -> source-file mapping for every composed figure."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brainturtle import config as cfg

PANEL_SOURCES = {
    "fig1a": ("age_shap_ranking.csv", "data/ (age SHAP ranking)"),
    "fig1b": ("fig1_suppression.csv", "analyses/a_suppression.py"),
    "fig1c": ("fig1_suppression.csv", "analyses/a_suppression.py"),
    "fig2a": ("fig2a_removal_ladder.csv", "analyses/b_removal_sweep.py"),
    "fig2b": ("fig2b_null_draws.csv", "analyses/c_removal_null.py"),
    "fig2c": ("fig2c_agebins.csv", "analyses/d_agebins.py"),
    "fig3": ("fig3_category_rates.csv", "analyses/e_category.py"),
    "fig4a": ("fig4_external.csv", "analyses/f_external.py"),
    "fig4b": ("fig4_external.csv", "analyses/f_external.py"),
    "suppS1": ("fig2a_removal_ladder.csv", "analyses/b_removal_sweep.py"),
    "suppS5": ("calibration.csv", "analyses/i_calibration.py"),
    "suppS9": ("suppfig9_null_draws.csv", "analyses/g_gender_control.py"),
}

KANG_PANEL_COHORT = {
    "fig4a": "Kang",
    "suppS4": "Kang (investigator-removed)",
}


def resolve(panel, n_perm=None):
    """Absolute path to a panel's source. Raises if it is not there."""
    if panel not in PANEL_SOURCES:
        raise KeyError(f"unknown panel {panel!r}; known: "
                       f"{sorted(PANEL_SOURCES)}")
    name, producer = PANEL_SOURCES[panel]
    if panel == "fig1a":
        path = cfg.DATA / name
    else:
        path = cfg.RESULTS / f"n{n_perm or cfg.N_PERM}" / name
    if not path.exists():
        raise FileNotFoundError(
            f"panel {panel} needs {path}, which does not exist. "
            f"Run {producer} first. "
            "Not falling back to another file -- that is how Figure 4a "
            "acquired the wrong Kang curve."
        )
    return path


def check_all(n_perm=None):
    """State of every panel source. Run before regenerating any figure."""
    rows = []
    for panel in PANEL_SOURCES:
        try:
            p = resolve(panel, n_perm)
            rows.append((panel, "ok", str(p)))
        except FileNotFoundError as e:
            rows.append((panel, "MISSING", str(e).split(", which")[0]))
    return rows


if __name__ == "__main__":
    print(f"panel sources for N_PERM={cfg.N_PERM}\n")
    width = max(len(p) for p in PANEL_SOURCES)
    missing = 0
    for panel, state, detail in check_all():
        flag = " " if state == "ok" else "!"
        print(f" {flag} {panel:<{width}}  {state:8s} {detail}")
        missing += state != "ok"
    print()
    for panel, cohort in KANG_PANEL_COHORT.items():
        print(f"   {panel} must plot cohort: {cohort!r}")
    if missing:
        print(f"\n {missing} panel source(s) missing -- run the listed "
              "analyses before regenerating figures.")
