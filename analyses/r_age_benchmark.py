"""Age benchmark and incremental value over age."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brainturtle import config as cfg
from brainturtle import data as bt_data
from brainturtle import evaluate, featuresets


def speaker_table(keep_feats, name):
    """Speaker-averaged out-of-fold probability for one feature set."""
    _, _, frames = evaluate.pooled_predictions(keep_feats, with_age=True)
    g = (frames.groupby(["speaker_id", "label"], as_index=False)
                .agg(prob=("prob", "mean"), entryage=("entryage", "first")))
    return g.rename(columns={"prob": name})


def age_only_predictions():
    """Out-of-fold logistic on age alone, using the identical splits."""
    df = bt_data.pitt_with_age()
    rows = []
    for seed in cfg.SEEDS:
        trm, vam = evaluate.split_masks(seed, with_age=True)
        tr = df.loc[trm & df["entryage"].notna()]
        va = df.loc[vam & df["entryage"].notna()]
        if tr.empty or va.empty:
            continue
        clf = LogisticRegression()
        clf.fit(tr[["entryage"]].values, tr["label"].values)
        p = clf.predict_proba(va[["entryage"]].values)[:, 1]
        f = va[["speaker_id", "label", "entryage"]].copy()
        f["prob"] = p
        rows.append(f)
    frames = pd.concat(rows, ignore_index=True)
    pooled = roc_auc_score(frames["label"], frames["prob"])
    g = (frames.groupby(["speaker_id", "label"], as_index=False)
                .agg(age_only=("prob", "mean"), entryage=("entryage", "first")))
    return g, pooled


def boot_indices(n, n_boot, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def boot_auc(y, p, idx):
    """AUC on each resample; draws with only one class present are dropped."""
    out = np.empty(len(idx))
    out[:] = np.nan
    for i, ix in enumerate(idx):
        yy = y[ix]
        if yy.min() == yy.max():
            continue
        out[i] = roc_auc_score(yy, p[ix])
    return out


def ci(v):
    v = v[~np.isnan(v)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def diff_test(a, b):
    """Paired bootstrap difference a - b: point CI and two-sided p."""
    d = a - b
    d = d[~np.isnan(d)]
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return lo, hi, float(min(1.0, max(p, 1.0 / len(d))))


def cv_scores(X, y, seed=cfg.BOOT_SEED):
    """Out-of-fold probabilities from a repeated-CV logistic, averaged."""
    cvk = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=seed)
    acc = np.zeros(len(y))
    cnt = np.zeros(len(y))
    for tr, va in cvk.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000)
        clf.fit(sc.transform(X[tr]), y[tr])
        acc[va] += clf.predict_proba(sc.transform(X[va]))[:, 1]
        cnt[va] += 1
    return acc / np.maximum(cnt, 1)


def main():
    outdir = cfg.RESULTS / f"n{cfg.N_PERM}"
    outdir.mkdir(parents=True, exist_ok=True)

    feats = featuresets.common_features()
    keep0 = feats
    keep30 = featuresets.keep_set(cfg.PRIMARY_THRESHOLD, feats)
    featuresets.assert_matches_models(keep30, cfg.PRIMARY_THRESHOLD)
    print(f"  {len(feats)} features; {len(keep30)} kept at "
          f"{cfg.PRIMARY_THRESHOLD:.0%} removal")

    s0 = speaker_table(keep0, "drop0")
    s30 = speaker_table(keep30, "drop30")
    sage, age_pooled = age_only_predictions()

    df = (s0.merge(s30[["speaker_id", "label", "drop30"]],
                   on=["speaker_id", "label"])
            .merge(sage[["speaker_id", "label", "age_only"]],
                   on=["speaker_id", "label"]))
    df = df.dropna(subset=["entryage"]).reset_index(drop=True)
    print(f"  {len(df)} held-out speakers with a recorded age "
          f"({int(df['label'].sum())} AD)")

    y = df["label"].values.astype(int)
    idx = boot_indices(len(df), cfg.N_BOOT, cfg.BOOT_SEED)

    aucs, boots = {}, {}
    for col in ("age_only", "drop0", "drop30"):
        p = df[col].values
        aucs[col] = roc_auc_score(y, p)
        boots[col] = boot_auc(y, p, idx)

    gap_ad = df.loc[y == 1, "entryage"].mean()
    gap_ct = df.loc[y == 0, "entryage"].mean()

    rows = [
        dict(quantity="age gap AD - control (years)",
             value=gap_ad - gap_ct, lo=np.nan, hi=np.nan, p=np.nan,
             detail=f"AD {gap_ad:.2f} +/- {df.loc[y == 1, 'entryage'].std():.2f}; "
                    f"control {gap_ct:.2f} +/- "
                    f"{df.loc[y == 0, 'entryage'].std():.2f}"),
    ]
    for col, label in (("age_only", "age alone"),
                       ("drop0", "0% removal (full feature space)"),
                       ("drop30", "30% removal")):
        lo, hi = ci(boots[col])
        rows.append(dict(quantity=f"AUC, {label}", value=aucs[col],
                         lo=lo, hi=hi, p=np.nan, detail="speaker-averaged, P2"))
    for a, b, label in (("drop0", "age_only", "0% model vs age"),
                        ("drop30", "age_only", "30% model vs age"),
                        ("drop30", "drop0", "30% vs 0% model")):
        lo, hi, p = diff_test(boots[a], boots[b])
        rows.append(dict(quantity=f"AUC difference, {label}",
                         value=aucs[a] - aucs[b], lo=lo, hi=hi, p=p,
                         detail="paired bootstrap over speakers"))
    rows.append(dict(quantity="AUC, age alone (pooled out-of-fold rows, P1)",
                     value=age_pooled, lo=np.nan, hi=np.nan, p=np.nan,
                     detail="reference only; P2 is the conservative figure"))
    bench = pd.DataFrame(rows)
    bench.to_csv(outdir / "fig1_age_benchmark.csv", index=False)

    roc_rows = []
    for col, label in (("age_only", "Age alone"),
                       ("drop0", "0% removal"),
                       ("drop30", "30% removal")):
        fpr, tpr, _ = roc_curve(y, df[col].values)
        roc_rows.append(pd.DataFrame({"model": label, "fpr": fpr, "tpr": tpr,
                                      "auc": aucs[col]}))
    pd.concat(roc_rows, ignore_index=True).to_csv(
        outdir / "fig1_age_roc.csv", index=False)

    df[["speaker_id", "label", "entryage", "age_only", "drop0",
        "drop30"]].to_csv(outdir / "fig1_age_distribution.csv", index=False)

    age = df[["entryage"]].values
    models = {
        "age alone": age,
        "age + undeconfounded score": np.column_stack(
            [age[:, 0], df["drop0"].values]),
        "age + deconfounded score": np.column_stack(
            [age[:, 0], df["drop30"].values]),
    }
    oof = {k: cv_scores(X, y) for k, X in models.items()}
    inc_auc = {k: roc_auc_score(y, v) for k, v in oof.items()}
    inc_boot = {k: boot_auc(y, v, idx) for k, v in oof.items()}

    base = inc_boot["age alone"]
    d_un = inc_boot["age + undeconfounded score"] - base
    d_de = inc_boot["age + deconfounded score"] - base
    lo_un, hi_un = ci(d_un)
    lo_de, hi_de = ci(d_de)
    lo_d, hi_d, p_d = diff_test(inc_boot["age + deconfounded score"],
                                inc_boot["age + undeconfounded score"])

    inc_rows = []
    for k in models:
        lo, hi = ci(inc_boot[k])
        inc_rows.append(dict(model=k, auc=inc_auc[k], lo=lo, hi=hi,
                             increment=inc_auc[k] - inc_auc["age alone"],
                             inc_lo=np.nan, inc_hi=np.nan, p=np.nan))
    inc_rows[1].update(inc_lo=lo_un, inc_hi=hi_un)
    inc_rows[2].update(inc_lo=lo_de, inc_hi=hi_de)
    inc = pd.DataFrame(inc_rows)
    inc.loc[len(inc)] = dict(
        model="difference of increments (deconfounded - undeconfounded)",
        auc=np.nan, lo=np.nan, hi=np.nan,
        increment=inc_auc["age + deconfounded score"]
                  - inc_auc["age + undeconfounded score"],
        inc_lo=lo_d, inc_hi=hi_d, p=p_d)
    inc.to_csv(outdir / "fig4_incremental.csv", index=False)

    def partial_rho(score, ages, labels):
        rs = pd.Series(score).rank().values
        ra = pd.Series(ages).rank().values
        X = np.column_stack([np.ones(len(labels)), labels])
        e1 = rs - X @ np.linalg.lstsq(X, rs, rcond=None)[0]
        e2 = ra - X @ np.linalg.lstsq(X, ra, rcond=None)[0]
        return float(np.corrcoef(e1, e2)[0, 1])

    def plain_rho(a, b):
        return float(np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0, 1])

    ages = df["entryage"].values
    ctrl = df[df["label"] == 0]
    rho_rows, boot_partial = [], {}
    for col, label in (("drop0", "0% removal"), ("drop30", "30% removal")):
        point = partial_rho(df[col].values, ages, y)
        draws = []
        for ix in idx:
            s = df.iloc[ix]
            lab = s["label"].values.astype(int)
            if lab.min() == lab.max():
                continue
            draws.append(partial_rho(s[col].values, s["entryage"].values, lab))
        draws = np.asarray(draws)
        boot_partial[col] = draws
        lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
        rho_rows.append(dict(quantity=f"partial rho given label, {label}",
                             value=point, lo=lo, hi=hi, p=np.nan, n=len(df)))
        rho_rows.append(dict(quantity=f"rho within controls, {label}",
                             value=plain_rho(ctrl[col].values,
                                             ctrl["entryage"].values),
                             lo=np.nan, hi=np.nan, p=np.nan, n=len(ctrl)))
    d = boot_partial["drop30"] - boot_partial["drop0"]
    rho_rows.append(dict(
        quantity="difference in partial rho (30% - 0%)",
        value=partial_rho(df["drop30"].values, ages, y)
              - partial_rho(df["drop0"].values, ages, y),
        lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)),
        p=float(2 * min((d <= 0).mean(), (d >= 0).mean())), n=len(df)))

    for n, what in ((len(df), "all speakers"), (len(ctrl), "controls only")):
        z = 1.959963985 + 0.8416212336
        rho_rows.append(dict(quantity=f"detectable |rho| at 80% power, {what}",
                             value=float(np.tanh(z / np.sqrt(n - 3))),
                             lo=np.nan, hi=np.nan, p=np.nan, n=n))

    from scipy import stats

    def loglik(cols):
        X = df[cols].values.astype(float)
        X = (X - X.mean(0)) / X.std(0)
        m = LogisticRegression(C=1e6, max_iter=5000).fit(X, y)
        p = np.clip(m.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
        return float((y * np.log(p) + (1 - y) * np.log(1 - p)).sum())

    for col, label in (("drop0", "undeconfounded"), ("drop30", "deconfounded")):
        both, only_score, only_age = (loglik(["entryage", col]), loglik([col]),
                                      loglik(["entryage"]))
        for chi2, what in ((2 * (both - only_score), f"age over the {label} score"),
                           (2 * (both - only_age), f"{label} score over age")):
            rho_rows.append(dict(quantity=f"likelihood-ratio chi2(1), {what}",
                                 value=float(chi2), lo=np.nan, hi=np.nan,
                                 p=float(stats.chi2.sf(chi2, 1)), n=len(df)))

    rho = pd.DataFrame(rho_rows)
    rho.to_csv(outdir / "fig4_score_age_association.csv", index=False)
    print(rho.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    pd.set_option("display.width", 200)
    print(bench.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(inc.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n  -> {outdir}/fig1_age_benchmark.csv, fig1_age_roc.csv, "
          f"fig1_age_distribution.csv, fig4_incremental.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
