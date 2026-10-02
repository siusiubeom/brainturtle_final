"""Model training, splitting, and prediction pooling."""
import functools
import warnings

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from . import config as cfg
from . import data as bt_data

warnings.filterwarnings("ignore", message=".*valid feature names.*")


@functools.lru_cache(maxsize=None)
def split_masks(seed, with_age=False):
    """Speaker-level stratified split. Identical across every analysis."""
    df = bt_data.pitt_with_age() if with_age else bt_data.pitt()
    speakers = df[["speaker_id", "label"]].drop_duplicates()
    tr, va = train_test_split(speakers, test_size=cfg.TEST_SIZE,
                              stratify=speakers["label"], random_state=seed)
    return (df["speaker_id"].isin(set(tr["speaker_id"])).values,
            df["speaker_id"].isin(set(va["speaker_id"])).values)


KNOWN_SPLIT_NOTE = """\
Speaker 172 is a longitudinal converter: control at baseline (172-0.wav, in
pitt_control_cookie) and AD at follow-up (172-1/2/3.wav, in
pitt_dementia_cookie). The data is correct -- DementiaBank Pitt follows
participants over time and this participant converted.

Speaker-level aggregation therefore yields two rows for 172, and because the
split de-duplicates on (speaker_id, label) they become two split units. The
same person consequently appears in both train and validation in seeds 1 and 4
(not in 0, 2, 3).

This is left as-is by decision: it affects 1 of 292 speakers, and changing the
split unit re-partitions all five folds and moves every published number.
Report it in Limitations rather than silently altering the protocol.

Note that "speaker-level splitting was applied consistently across all corpora"
in the Reproducibility section is therefore not strictly true for this one
participant.
"""


def _classifier(seed):
    return lgb.LGBMClassifier(class_weight="balanced", random_state=seed,
                              verbose=-1, n_jobs=cfg.MODEL_THREADS)


def pooled_predictions(keep_feats, seeds=None, with_age=False):
    """Train per seed, predict that seed's held-out speakers, concatenate."""
    seeds = cfg.SEEDS if seeds is None else seeds
    df = bt_data.pitt_with_age() if with_age else bt_data.pitt()
    keep_feats = list(keep_feats)
    ys, ps, frames = [], [], []
    for seed in seeds:
        trm, vam = split_masks(seed, with_age)
        clf = _classifier(seed)
        clf.fit(df.loc[trm, keep_feats].values, df.loc[trm, "label"].values)
        prob = clf.predict_proba(df.loc[vam, keep_feats].values)[:, 1]
        cols = ["speaker_id", "label"] + (["entryage"] if with_age else [])
        f = df.loc[vam, cols].copy()
        f["prob"] = prob
        f["seed"] = seed
        ys.append(df.loc[vam, "label"].values)
        ps.append(prob)
        frames.append(f)
    return (np.concatenate(ys), np.concatenate(ps),
            pd.concat(frames, ignore_index=True))


def pooled_auc(keep_feats, seeds=None):
    y, p, _ = pooled_predictions(keep_feats, seeds)
    return roc_auc_score(y, p)


def isolation_forest_auc(keep_feats, seeds=None):
    """Suppression-test scorer (Figure 1). Anomaly score = -decision_function."""
    seeds = cfg.SEEDS if seeds is None else seeds
    df = bt_data.pitt()
    keep_feats = list(keep_feats)
    ys, ps = [], []
    for seed in seeds:
        trm, vam = split_masks(seed)
        m = IsolationForest(random_state=seed, contamination="auto",
                            n_jobs=cfg.MODEL_THREADS)
        m.fit(df.loc[trm, keep_feats].values)
        ps.append(-m.decision_function(df.loc[vam, keep_feats].values))
        ys.append(df.loc[vam, "label"].values)
    return roc_auc_score(np.concatenate(ys), np.concatenate(ps))


def external_probability_matrix(df, pct=None, seeds=None):
    """Score the saved Pitt models on an external cohort."""
    pct = cfg.PRIMARY_THRESHOLD if pct is None else pct
    seeds = cfg.SEEDS if seeds is None else seeds
    pct_int = int(round(pct * 100))
    cols = []
    for seed in seeds:
        model_p = cfg.MODEL_DIR / f"pitt_drop{pct_int}_seed{seed}.pkl"
        feat_p = cfg.FEAT_DIR / f"pitt_drop{pct_int}_seed{seed}_feats.pkl"
        if not model_p.exists() or not feat_p.exists():
            raise FileNotFoundError(f"missing {model_p.name} or {feat_p.name}")
        clf = joblib.load(model_p)
        feats = joblib.load(feat_p)
        missing = [f for f in feats if f not in df.columns]
        if missing:
            raise ValueError(
                f"external cohort lacks {len(missing)} features required by "
                f"drop{pct_int}_seed{seed}, e.g. {missing[:5]}")
        cols.append(clf.predict_proba(df[list(feats)].values)[:, 1])
    return np.column_stack(cols), df["label"].astype(int).to_numpy()


def train_and_save(keep_feats, pct, seeds=None):
    """Deliberately (re)train and persist the drop-`pct` models."""
    seeds = cfg.SEEDS if seeds is None else seeds
    df = bt_data.pitt()
    pct_int = int(round(pct * 100))
    cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cfg.FEAT_DIR.mkdir(parents=True, exist_ok=True)
    keep_feats = list(keep_feats)
    for seed in seeds:
        trm, _ = split_masks(seed)
        clf = _classifier(seed)
        clf.fit(df.loc[trm, keep_feats].values, df.loc[trm, "label"].values)
        joblib.dump(clf, cfg.MODEL_DIR / f"pitt_drop{pct_int}_seed{seed}.pkl")
        joblib.dump(keep_feats, cfg.FEAT_DIR / f"pitt_drop{pct_int}_seed{seed}_feats.pkl")
