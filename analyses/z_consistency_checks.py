"""Every provenance guard in one place. Run before regenerating anything."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from brainturtle import config as cfg, data, featuresets as fs, report

failures = []


def check(name, fn):
    try:
        detail = fn()
        print(f"  PASS  {name}")
        if detail:
            for line in str(detail).splitlines():
                print(f"          {line}")
    except Exception as e:
        failures.append(name)
        print(f"  FAIL  {name}")
        for line in str(e).splitlines():
            print(f"          {line}")


report.header("Consistency checks")

check("drop30 feature set matches saved models",
      lambda: f"{len(fs.keep_set(cfg.PRIMARY_THRESHOLD))} features kept"
              if fs.assert_matches_models(fs.keep_set(cfg.PRIMARY_THRESHOLD))
              else None)


def legacy_must_differ():
    feats = fs.common_features()
    legacy = set(fs.legacy_union_drop_set(feats))
    keep = [f for f in feats if f not in legacy]
    try:
        fs.assert_matches_models(keep)
    except AssertionError:
        return (f"legacy union rule gives {len(legacy)} features, canonical "
                f"gives {len(fs.drop_set())} -- correctly distinct")
    raise AssertionError("legacy union rule now matches the saved models; "
                         "the canonical definition has drifted")


check("legacy union rule is still distinct from canonical", legacy_must_differ)


def ranking_canonical():
    """age_shap_ranking.csv is the canonical ranking."""
    fs.assert_matches_models(fs.keep_set(cfg.PRIMARY_THRESHOLD))
    c = fs.compare_rankings()
    note = (f"canonical = {fs.CANONICAL_RANKING} (mean |SHAP|, matches models)\n"
            f"superseded = {fs.ALTERNATE_RANKING} (LightGBM gain, no generator "
            f"in repo)\n"
            f"they share {c['overlap'].iloc[3]['overlap']}/33 at the cut, so "
            f"anything built from the\nsuperseded file must be regenerated -- "
            f"Figure 1 in particular.")
    return note


check("age ranking decision is enforced", ranking_canonical)


def orphans():
    suspects = ["gender_sensitive_features_to_drop.csv",
                "age_gender_sensitive_features_to_drop.csv",
                "keep_features_age_gender_deconfounded.csv",
                "age_sensitive_features_to_drop.csv"]
    present = [s for s in suspects if (cfg.DATA / s).exists()]
    if present:
        raise AssertionError(
            "these files are written by the notebooks but read by nothing in\n"
            "the analysis path; anything built from them is unverified:\n  "
            + "\n  ".join(present))
    return None


check("no orphan feature-list CSVs in the analysis path", orphans)


def external_features():
    from brainturtle import evaluate
    out = []
    for name, _ in data.external_cohorts():
        df = data.external(name)
        evaluate.external_probability_matrix(df, cfg.PRIMARY_THRESHOLD)
        out.append(f"{name}: {len(df)} speakers, all required features present")
    return "\n".join(out)


check("external cohorts have all required features", external_features)


def p_resolution():
    two = cfg.p_floor(two_sided=True)
    if two > 0.001:
        raise AssertionError(
            f"N_PERM={cfg.N_PERM} gives a two-sided p floor of {two:.2g}. "
            "You cannot report p<0.001 or plot *** at this size; "
            "N_PERM>=1999 is required.")
    return f"two-sided floor {two:.2g}, one-sided {cfg.p_floor(two_sided=False):.2g}"


check("N_PERM supports the p-values being reported", p_resolution)

print()
if failures:
    print(f"  {len(failures)} check(s) failed: {', '.join(failures)}")
    sys.exit(1)
print("  all checks passed")
