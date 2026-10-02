"""Every resampling procedure in the paper, defined exactly once."""
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from . import config as cfg


def permutation_p(null, observed, two_sided=True):
    """Add-one empirical p. Never returns 0."""
    null = np.asarray(null, float)
    n = null.size
    hi = (1 + int((null >= observed).sum())) / (n + 1)
    lo = (1 + int((null <= observed).sum())) / (n + 1)
    return min(1.0, 2 * min(hi, lo)) if two_sided else hi


def format_p(p, n_perm=None, two_sided=True):
    """Render a p at the resolution N_PERM supports, as "p < floor" below it."""
    floor = cfg.p_floor(n_perm, two_sided)
    if p <= floor:
        return f"p < {floor:.2g}"
    return f"p = {p:.4g}"


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def bootstrap_auc(y, p, n_boot=None, seed=None):
    """Percentile CI for a single AUC. Resamples rows."""
    n_boot = cfg.N_BOOT if n_boot is None else n_boot
    seed = cfg.BOOT_SEED if seed is None else seed
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    out = []
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        out.append(roc_auc_score(y[s], p[s]))
    a = np.asarray(out, float)
    if a.size == 0:
        return np.nan, np.nan, np.nan, a
    return (float(a.mean()),
            float(np.percentile(a, cfg.CI_LOW)),
            float(np.percentile(a, cfg.CI_HIGH)),
            a)


def bootstrap_auc_difference(y, p1, p2, n_boot=None, seed=None):
    """Paired case-resampling test for AUC(p1) - AUC(p2)."""
    n_boot = cfg.N_BOOT if n_boot is None else n_boot
    seed = cfg.BOOT_SEED if seed is None else seed
    y, p1, p2 = np.asarray(y), np.asarray(p1), np.asarray(p2)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    d = []
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        d.append(roc_auc_score(y[s], p1[s]) - roc_auc_score(y[s], p2[s]))
    d = np.asarray(d, float)
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    if sd == 0:
        z = np.nan
        p_wald = 1.0 if mean == 0 else 0.0
    else:
        z = mean / sd
        p_wald = float(2 * (1 - norm.cdf(abs(z))))
    p_pct = float(2 * min((d <= 0).mean(), (d >= 0).mean()))
    return {
        "mean_diff": mean, "sd": sd, "z": z,
        "ci_low": float(np.percentile(d, cfg.CI_LOW)),
        "ci_high": float(np.percentile(d, cfg.CI_HIGH)),
        "p_wald": p_wald,
        "p_percentile": p_pct,
        "p_percentile_floor": 2.0 / len(d),
        "n_boot_effective": len(d),
        "diffs": d,
    }


def external_bootstrap_auc(P, y, n_boot=None, seed=None):
    """Speaker-level bootstrap for external validation."""
    n_boot = cfg.N_BOOT if n_boot is None else n_boot
    seed = cfg.BOOT_SEED if seed is None else seed
    P, y = np.asarray(P, float), np.asarray(y)
    n = len(y)
    if P.shape[0] != n:
        raise ValueError(
            f"P is {P.shape}; expected ({n}, n_seeds). Seeds must be columns "
            "-- np.column_stack, not np.concatenate."
        )
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        out.append(roc_auc_score(yb, P[idx].mean(axis=1)))
    a = np.asarray(out, float)
    if a.size == 0:
        return np.nan, np.nan, np.nan, a
    return (float(a.mean()),
            float(np.percentile(a, cfg.CI_LOW)),
            float(np.percentile(a, cfg.CI_HIGH)),
            a)


def label_permutation_null(y, p, n_perm=None, seed=None):
    """Null AUC distribution from shuffling labels against fixed predictions."""
    n_perm = cfg.N_PERM if n_perm is None else n_perm
    seed = cfg.PERM_SEED if seed is None else seed
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    return np.array([roc_auc_score(rng.permutation(y), p) for _ in range(n_perm)])


def subset_permutation_null(evaluate, feats, k, n_perm=None, seed=None,
                            n_jobs=None):
    """Null from drawing `n_perm` independent random feature subsets of size k."""
    from joblib import Parallel, delayed
    import os
    n_perm = cfg.N_PERM if n_perm is None else n_perm
    seed = cfg.PERM_SEED if seed is None else seed
    n_jobs = int(os.environ.get("NJOBS", -1)) if n_jobs is None else n_jobs
    feats = list(feats)

    def one(i):
        rng = np.random.default_rng(seed + i)
        return evaluate(list(rng.choice(feats, size=k, replace=False)))

    batch = int(os.environ.get("BATCH", 0)) or max(
        1, min(256, n_perm // (abs(n_jobs) * 4) or 1))
    return np.array(Parallel(n_jobs=n_jobs, batch_size=batch)(
        delayed(one)(i) for i in range(n_perm)))


def stability_sweep(make_null, observed, seed_bases=None, n_perm=None,
                    two_sided=True):
    """Repeat a permutation analysis under several independent RNG seeds."""
    seed_bases = cfg.PERM_SEED_BASES if seed_bases is None else seed_bases
    n_perm = cfg.N_PERM_STABILITY if n_perm is None else n_perm
    rows = []
    for base in seed_bases:
        null = make_null(cfg.PERM_SEED + base * 1_000_003, n_perm)
        rows.append({
            "perm_seed_base": base,
            "null_mean": float(null.mean()),
            "null_sd": float(null.std(ddof=1)),
            "null_lo": float(np.percentile(null, cfg.CI_LOW)),
            "null_hi": float(np.percentile(null, cfg.CI_HIGH)),
            "n_ge_observed": int((null >= observed).sum()),
            "z": float((observed - null.mean()) / null.std(ddof=1)),
            "p": permutation_p(null, observed, two_sided),
        })
    return rows


def summarize_null(null, observed, n_perm=None, two_sided=True):
    null = np.asarray(null, float)
    p = permutation_p(null, observed, two_sided)
    return {
        "observed": float(observed),
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)),
        "null_lo": float(np.percentile(null, cfg.CI_LOW)),
        "null_hi": float(np.percentile(null, cfg.CI_HIGH)),
        "n_perm": int(null.size),
        "n_ge_observed": int((null >= observed).sum()),
        "z": float((observed - null.mean()) / null.std(ddof=1)),
        "p": p,
        "p_formatted": format_p(p, n_perm or null.size, two_sided),
        "p_floor": cfg.p_floor(n_perm or null.size, two_sided),
    }
