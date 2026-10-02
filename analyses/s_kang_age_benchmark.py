"""Age benchmark on the Kang Corpus."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brainturtle import config as cfg, data, evaluate, report

DEMO = cfg.ROOT / "demo-kang.xlsx"
THRESHOLDS = (0.0, 0.30, 0.35)

report.run_banner("s_kang_age_benchmark")


def load_demographics():
    """Speaker-level age, sex and CDR, keyed the way the feature table is."""
    raw = pd.read_excel(DEMO, sheet_name="data", header=None)
    hrow = next(i for i in raw.index
                if "ID" in [str(v).strip() for v in raw.loc[i].tolist()])
    demo = raw.iloc[hrow + 1:].copy()
    demo.columns = [str(v).strip() for v in raw.loc[hrow].tolist()]
    demo = demo.dropna(subset=["ID"])
    demo["speaker_id"] = demo["ID"].astype(int).map(lambda i: "%02d" % i)
    demo["age"] = demo["age"].astype(float)
    demo["female"] = demo["gender"].astype(str).str.upper().str[0].eq("F")
    return demo[["speaker_id", "age", "female", "CDR", "role"]]


def boot_indices(n, n_boot, seed):
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def boot_auc(y, p, idx):
    out = np.full(len(idx), np.nan)
    for i, ix in enumerate(idx):
        yy = y[ix]
        if yy.min() != yy.max():
            out[i] = roc_auc_score(yy, p[ix])
    return out


def ci(v):
    v = v[~np.isnan(v)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def diff_test(a, b):
    d = (a - b)[~np.isnan(a - b)]
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)),
            float(min(1.0, max(p, 1.0 / len(d)))))


def cv_scores(X, y, seed=cfg.BOOT_SEED):
    """Out-of-fold probabilities from a repeated-CV logistic, averaged."""
    cvk = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=seed)
    acc, cnt = np.zeros(len(y)), np.zeros(len(y))
    for tr, va in cvk.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000).fit(sc.transform(X[tr]), y[tr])
        acc[va] += clf.predict_proba(sc.transform(X[va]))[:, 1]
        cnt[va] += 1
    return acc / np.maximum(cnt, 1)


def partial_rho(score, ages, labels):
    """Spearman correlation of score and age after residualising on label."""
    rs = pd.Series(score).rank().values
    ra = pd.Series(ages).rank().values
    X = np.column_stack([np.ones(len(labels)), labels])
    e1 = rs - X @ np.linalg.lstsq(X, rs, rcond=None)[0]
    e2 = ra - X @ np.linalg.lstsq(X, ra, rcond=None)[0]
    return float(np.corrcoef(e1, e2)[0, 1])


def main():
    outdir = cfg.RESULTS / f"n{cfg.N_PERM}"
    outdir.mkdir(parents=True, exist_ok=True)

    kang = data.external("Kang")
    kang["speaker_id"] = kang["speaker_id"].astype(str).str.zfill(2)
    demo = load_demographics()

    df = kang[["speaker_id", "label"]].merge(demo, on="speaker_id", how="left")
    if df["age"].isna().any():
        raise RuntimeError("unmatched speakers: %s"
                           % df.loc[df["age"].isna(), "speaker_id"].tolist())
    mismatch = (df["label"] == 1) != (df["role"] == "MCI")
    if mismatch.any():
        raise RuntimeError("label/role disagree for %d speakers"
                           % int(mismatch.sum()))
    print(f"  {len(df)} speakers matched to demographics "
          f"({int(df['label'].sum())} MCI, {int((1 - df['label']).sum())} HC)")

    for pct in THRESHOLDS:
        P, y = evaluate.external_probability_matrix(kang, pct)
        df["drop%d" % round(pct * 100)] = P.mean(axis=1)
    y = df["label"].to_numpy().astype(int)
    idx = boot_indices(len(df), cfg.N_BOOT, cfg.BOOT_SEED)

    cols = ["age"] + ["drop%d" % round(p * 100) for p in THRESHOLDS]
    aucs, boots = {}, {}
    for c in cols:
        aucs[c] = roc_auc_score(y, df[c].values)
        boots[c] = boot_auc(y, df[c].values, idx)

    a_mci = df.loc[y == 1, "age"]
    a_hc = df.loc[y == 0, "age"]
    rows = [dict(
        quantity="age gap MCI - HC (years)", value=a_mci.mean() - a_hc.mean(),
        lo=np.nan, hi=np.nan, p=np.nan,
        detail=f"MCI {a_mci.mean():.2f} +/- {a_mci.std():.2f}; "
               f"HC {a_hc.mean():.2f} +/- {a_hc.std():.2f}")]

    labels = {"age": "age alone",
              "drop0": "0% removal (undeconfounded)",
              "drop30": "30% removal",
              "drop35": "35% removal"}
    for c in cols:
        lo, hi = ci(boots[c])
        rows.append(dict(quantity=f"AUC, {labels[c]}", value=aucs[c],
                         lo=lo, hi=hi, p=np.nan,
                         detail="speaker-averaged over five Pitt models"
                                if c != "age" else "no model; age as the score"))
    for a, b in (("drop0", "age"), ("drop30", "age"), ("drop35", "age")):
        lo, hi, p = diff_test(boots[a], boots[b])
        rows.append(dict(quantity=f"AUC difference, {labels[a]} vs age",
                         value=aucs[a] - aucs[b], lo=lo, hi=hi, p=p,
                         detail="paired bootstrap over speakers"))

    for c in ("drop0", "drop30", "drop35"):
        rows.append(dict(
            quantity=f"partial Spearman rho, {labels[c]} score vs age | label",
            value=partial_rho(df[c].values, df["age"].values, y),
            lo=np.nan, hi=np.nan, p=np.nan, detail="conditioning on diagnosis"))

    f_mci = int(df.loc[y == 1, "female"].sum())
    f_hc = int(df.loc[y == 0, "female"].sum())
    rows.append(dict(quantity="female fraction, MCI",
                     value=f_mci / int((y == 1).sum()), lo=np.nan, hi=np.nan,
                     p=np.nan, detail=f"{f_mci}/{int((y == 1).sum())}"))
    rows.append(dict(quantity="female fraction, HC",
                     value=f_hc / int((y == 0).sum()), lo=np.nan, hi=np.nan,
                     p=np.nan, detail=f"{f_hc}/{int((y == 0).sum())}"))

    age = df[["age"]].values
    models = {"age alone": age}
    for c in ("drop0", "drop30", "drop35"):
        models[f"age + {labels[c]} score"] = np.column_stack(
            [age[:, 0], df[c].values])
    oof = {k: cv_scores(X, y) for k, X in models.items()}
    inc_auc = {k: roc_auc_score(y, v) for k, v in oof.items()}
    inc_boot = {k: boot_auc(y, v, idx) for k, v in oof.items()}
    base = inc_boot["age alone"]
    for k in models:
        lo, hi = ci(inc_boot[k])
        d = dict(quantity=f"incremental AUC, {k}", value=inc_auc[k],
                 lo=lo, hi=hi, p=np.nan,
                 detail="5-fold x 20-repeat CV logistic, out-of-fold")
        if k != "age alone":
            dlo, dhi, dp = diff_test(inc_boot[k], base)
            d["detail"] += (f"; increment {inc_auc[k] - inc_auc['age alone']:+.4f} "
                            f"(95% CI {dlo:+.4f} to {dhi:+.4f}, p = {dp:.4f})")
        rows.append(d)

    inc_rows = []
    for k in models:
        lo, hi = ci(inc_boot[k])
        r = dict(model=k, auc=inc_auc[k], lo=lo, hi=hi,
                 increment=inc_auc[k] - inc_auc["age alone"],
                 inc_lo=np.nan, inc_hi=np.nan, p=np.nan)
        if k != "age alone":
            r["inc_lo"], r["inc_hi"], r["p"] = diff_test(inc_boot[k], base)
        else:
            r["increment"] = 0.0
        inc_rows.append(r)

    undec = "age + 0% removal (undeconfounded) score"
    dec = "age + 35% removal score"
    dlo, dhi, dp = diff_test(inc_boot[dec], inc_boot[undec])
    inc_rows.append(dict(
        model="difference of increments (deconfounded - undeconfounded)",
        auc=np.nan, lo=np.nan, hi=np.nan,
        increment=inc_auc[dec] - inc_auc[undec],
        inc_lo=dlo, inc_hi=dhi, p=dp))
    pd.DataFrame(inc_rows).to_csv(outdir / "fig4_kang_incremental.csv",
                                  index=False)

    res = pd.DataFrame(rows)
    res.to_csv(outdir / "kang_age_benchmark.csv", index=False)
    df.to_csv(outdir / "kang_speaker_table.csv", index=False)

    report.header("Kang Corpus age benchmark",
                  "demographics from Lee et al. 2025, Sci Rep 15, "
                  "doi:10.1038/s41598-025-14998-7")
    for r in rows:
        v = "" if pd.isna(r["value"]) else f"{r['value']:+.4f}"
        rng = "" if pd.isna(r["lo"]) else f"  [{r['lo']:+.4f}, {r['hi']:+.4f}]"
        pp = "" if pd.isna(r["p"]) else f"  p = {r['p']:.4f}"
        print(f"  {r['quantity']:58s} {v}{rng}{pp}")
    print()
    print(f"  wrote kang_age_benchmark.csv, kang_speaker_table.csv")


if __name__ == "__main__":
    main()
