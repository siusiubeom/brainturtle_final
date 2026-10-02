"""Corpus loading. One loader per cohort, used by every analysis and figure."""
import functools

import numpy as np
import pandas as pd

from . import config as cfg


@functools.lru_cache(maxsize=None)
def pitt():
    """Speaker-level Pitt features. 292 speakers, 110 shared features."""
    return pd.read_csv(cfg.FEATURES / "speaker_features.csv")


@functools.lru_cache(maxsize=None)
def commonvoice():
    return pd.read_csv(cfg.DATA / "commonvoice_speaker_features.csv")


def _speaker_to_int(s):
    """'002-0' -> 2. Pitt IDs carry a session suffix; ages are per speaker."""
    digits = "".join(ch for ch in str(s).split("-")[0] if ch.isdigit())
    return int(digits) if digits else pd.NA


@functools.lru_cache(maxsize=None)
def pitt_with_age():
    """Pitt features with `entryage` merged from the corpus metadata."""
    meta = pd.read_excel(cfg.DATA / "PItt-data.xlsx", sheet_name="data", header=2)
    meta.columns = meta.columns.astype(str).str.strip().str.lower()
    age = meta[["id", "entryage"]].copy()
    age["speaker_int"] = pd.to_numeric(age["id"], errors="coerce").astype("Int64")
    age["entryage"] = pd.to_numeric(age["entryage"], errors="coerce")
    age = age.dropna(subset=["speaker_int", "entryage"]).drop_duplicates("speaker_int")

    df = pitt().copy()
    df["speaker_int"] = df["speaker_id"].apply(_speaker_to_int).astype("Int64")
    return df.merge(age[["speaker_int", "entryage"]], on="speaker_int", how="left")


@functools.lru_cache(maxsize=None)
def commonvoice_gender():
    """Common Voice restricted to speakers with a usable binary gender label."""
    feats = commonvoice()
    meta = pd.read_csv(cfg.DATA / "commonvoice_metadata_clean.csv")
    g = meta[["client_id", "gender"]].dropna()
    g = g[g["gender"].isin({"male_masculine", "female_feminine"})]
    g = (g.groupby("client_id")["gender"]
           .agg(lambda s: s.value_counts().idxmax()).reset_index())
    g["gender_num"] = g["gender"].map({"male_masculine": 0, "female_feminine": 1})
    return feats.merge(g[["client_id", "gender_num"]], on="client_id", how="inner")


def external(name):
    """One external cohort by the name used in config.EXTERNAL_COHORTS."""
    spec = cfg.EXTERNAL_COHORTS[name]
    df = pd.read_csv(spec["path"])
    if "label" not in df.columns:
        raise ValueError(f"{name}: no label column in {spec['path']}")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    if df["label"].nunique() < 2:
        raise ValueError(f"{name}: only one class present, AUC undefined")
    return df


def external_cohorts(role=None):
    """Iterate cohorts, optionally filtered to 'primary' or 'sensitivity'."""
    for name, spec in cfg.EXTERNAL_COHORTS.items():
        if role is not None and spec["role"] != role:
            continue
        if not spec["path"].exists():
            continue
        yield name, spec


KANG_DEMOGRAPHICS = cfg.ROOT / "demo-kang.xlsx"


def kang_demographics():
    """Speaker-level age, sex and CDR for the Kang Corpus."""
    raw = pd.read_excel(KANG_DEMOGRAPHICS, sheet_name="data", header=None)
    hrow = next(i for i in raw.index
                if "ID" in [str(v).strip() for v in raw.loc[i].tolist()])
    d = raw.iloc[hrow + 1:].copy()
    d.columns = [str(v).strip() for v in raw.loc[hrow].tolist()]
    d = d.dropna(subset=["ID"])
    d["speaker_id"] = d["ID"].astype(int).map("{:02d}".format)
    d["age"] = d["age"].astype(float)
    d["female"] = d["gender"].astype(str).str.upper().str[0].eq("F")
    return d[["speaker_id", "age", "female", "CDR", "role"]]


def describe_cohorts():
    """Table 1 skeleton: participant characteristics per cohort."""
    rows = []
    p = pitt_with_age()
    rows.append({
        "cohort": "DementiaBank Pitt",
        "n_speakers": p["speaker_id"].nunique(),
        "n_ad": int((p["label"] == 1).sum()),
        "n_control": int((p["label"] == 0).sum()),
        "age_mean": float(p["entryage"].mean()),
        "age_sd": float(p["entryage"].std()),
        "age_note": "",
        "sex_note": "available in corpus metadata",
    })
    cv = commonvoice()
    agecol = "age_num_x" if "age_num_x" in cv.columns else "age_num"
    rows.append({
        "cohort": "Mozilla Common Voice",
        "n_speakers": len(cv),
        "n_ad": 0, "n_control": 0,
        "age_mean": float(cv[agecol].mean()),
        "age_sd": float(cv[agecol].std()),
        "age_note": "normative reference, no disease labels",
        "sex_note": "available",
    })
    kdemo = None
    for name, _ in external_cohorts():
        df = external(name)
        row = {
            "cohort": name,
            "n_speakers": len(df),
            "n_ad": int((df["label"] == 1).sum()),
            "n_control": int((df["label"] == 0).sum()),
            "age_mean": np.nan, "age_sd": np.nan,
            "age_note": ("matched by design (Luz et al. 2021)"
                         if "ADReSSo" in name else "not available"),
            "sex_note": ("matched by design (Luz et al. 2021)"
                         if "ADReSSo" in name else "not available"),
        }
        if name.startswith("Kang"):
            if kdemo is None:
                kdemo = kang_demographics()
            imp = kdemo[kdemo["role"] == "MCI"]
            ctl = kdemo[kdemo["role"] == "HC"]
            row["age_mean"] = float(kdemo["age"].mean())
            row["age_sd"] = float(kdemo["age"].std())
            row["age_note"] = (
                f"MCI {imp['age'].mean():.2f} +/- {imp['age'].std():.2f}; "
                f"HC {ctl['age'].mean():.2f} +/- {ctl['age'].std():.2f}; "
                f"gap {imp['age'].mean() - ctl['age'].mean():+.2f} y")
            row["sex_note"] = (
                f"female {int(imp['female'].sum())}/{len(imp)} MCI, "
                f"{int(ctl['female'].sum())}/{len(ctl)} HC")
        rows.append(row)
    return pd.DataFrame(rows)


def class_balance(y):
    """Base rate and the Brier score of an uninformative predictor."""
    y = np.asarray(y)
    rate = float(y.mean())
    return {"base_rate": rate, "brier_uninformative": rate * (1 - rate)}
