"""Canonical definition of the age-sensitive feature sets."""
import joblib
import pandas as pd

from . import config as cfg


CANONICAL_RANKING = "age_shap_ranking.csv"
ALTERNATE_RANKING = "age_feature_importance_commonvoice.csv"


def load_shap_ranking(path=None):
    """Age SHAP ranking, highest importance first. Defines every threshold."""
    df = pd.read_csv(cfg.DATA / (path or CANONICAL_RANKING))
    value_col = [c for c in df.columns if c != "feature"][0]
    return (df.sort_values(value_col, ascending=False)
              .reset_index(drop=True))


def compare_rankings():
    """How far apart are the two age rankings sitting in data/?"""
    a = load_shap_ranking(CANONICAL_RANKING)["feature"].tolist()
    b = load_shap_ranking(ALTERNATE_RANKING)["feature"].tolist()
    rows = []
    for k in (5, 11, 22, 33, 44):
        rows.append({"top_k": k,
                     "overlap": len(set(a[:k]) & set(b[:k])),
                     "agreement": len(set(a[:k]) & set(b[:k])) / k})
    return {"identical_order": a == b, "overlap": pd.DataFrame(rows),
            "canonical_top10": a[:10], "alternate_top10": b[:10]}


def _constant_columns(df):
    """Numeric columns that take a single value across every row."""
    num = df.select_dtypes("number")
    return {c for c in num.columns if num[c].nunique(dropna=False) <= 1}


def common_features():
    """The 110 features shared by Pitt and Common Voice."""
    ad = pd.read_csv(cfg.FEATURES / "speaker_features.csv")
    cv = pd.read_csv(cfg.DATA / "commonvoice_speaker_features.csv")
    ad_meta = {"speaker_id", "label", "split"}
    cv_meta = {"client_id", "age_num"}
    return sorted((set(c for c in ad.columns if c not in ad_meta)
                   & set(c for c in cv.columns if c not in cv_meta))
                  - _constant_columns(ad) - _constant_columns(cv))


def drop_set(pct=None, feats=None):
    """Rule B: the top `pct` of the SHAP age ranking. THE canonical rule."""
    pct = cfg.PRIMARY_THRESHOLD if pct is None else pct
    feats = common_features() if feats is None else feats
    k = int(len(feats) * pct)
    ranked = [f for f in load_shap_ranking()["feature"] if f in set(feats)]
    return ranked[:k]


def keep_set(pct=None, feats=None):
    feats = common_features() if feats is None else feats
    dropped = set(drop_set(pct, feats))
    return [f for f in feats if f not in dropped]


def legacy_union_drop_set(feats=None):
    """Rule A. Reproducible, but never label its output with a % threshold."""
    feats = common_features() if feats is None else feats
    from scipy.stats import spearmanr
    cv = pd.read_csv(cfg.DATA / "commonvoice_speaker_features.csv")
    age = cv["age_num_x"].values
    by_rho = {f for f in feats
              if abs(spearmanr(cv[f].values, age,
                               nan_policy="omit").statistic) >= cfg.LEGACY_UNION_RHO}
    ranked = [f for f in load_shap_ranking()["feature"] if f in set(feats)]
    by_shap = set(ranked[:int(len(feats) * cfg.LEGACY_UNION_SHAP_PCT)])
    return sorted(by_rho | by_shap)


def saved_model_features(pct=None, seed=0):
    """The feature list actually baked into a saved model pickle."""
    pct = cfg.PRIMARY_THRESHOLD if pct is None else pct
    p = cfg.FEAT_DIR / f"pitt_drop{int(round(pct * 100))}_seed{seed}_feats.pkl"
    return list(joblib.load(p))


def assert_matches_models(keep, pct=None, seeds=None):
    """Fail loudly if `keep` differs from what the saved models were fit on."""
    pct = cfg.PRIMARY_THRESHOLD if pct is None else pct
    seeds = cfg.SEEDS if seeds is None else seeds
    keep = list(keep)
    for seed in seeds:
        saved = saved_model_features(pct, seed)
        if list(saved) != keep:
            only_here = sorted(set(keep) - set(saved))
            only_saved = sorted(set(saved) - set(keep))
            raise AssertionError(
                f"feature set disagrees with pitt_drop{int(round(pct*100))}"
                f"_seed{seed}: {len(keep)} vs {len(saved)} features.\n"
                f"  only in candidate: {only_here[:8]}\n"
                f"  only in model    : {only_saved[:8]}\n"
                "If you meant the legacy union rule, call "
                "legacy_union_drop_set() and do not label it with a %."
            )
    return True


def categorize(f):
    """Acoustic family. Same rule the notebooks used, in one place now."""
    if f.startswith("mfcc"):
        return "MFCC"
    if "centroid" in f or "rolloff" in f or "zcr" in f:
        return "Spectral"
    if "pitch" in f:
        return "Pitch"
    if "rms" in f or "energy" in f:
        return "Energy"
    if "jitter" in f or "shimmer" in f or "hnr" in f:
        return "Voice Quality"
    return "Other"


def mfcc_subfamily(f):
    """shape (skew/kurt) vs non-shape."""
    if not f.startswith("mfcc"):
        return None
    return "shape" if f.endswith(("_skew", "_kurt")) else "non-shape"
