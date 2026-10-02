"""Figure 2b — is the 30% gain specific to WHICH features were removed?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from brainturtle import config as cfg, evaluate, featuresets as fs
from brainturtle import report, resampling as rs

report.run_banner("c_removal_null")

feats = fs.common_features()
k_drop = int(len(feats) * cfg.PRIMARY_THRESHOLD)

auc_baseline = evaluate.pooled_auc(feats)
keep30 = fs.keep_set(cfg.PRIMARY_THRESHOLD, feats)
fs.assert_matches_models(keep30)
auc_targeted = evaluate.pooled_auc(keep30)
observed = auc_targeted - auc_baseline


def evaluate_removal(drop_feats):
    keep = [f for f in feats if f not in set(drop_feats)]
    return evaluate.pooled_auc(keep) - auc_baseline


def make_null(seed, n_perm):
    return rs.subset_permutation_null(evaluate_removal, feats, k_drop,
                                      n_perm=n_perm, seed=seed)


print(f"  baseline AUC {auc_baseline:.4f} -> targeted AUC {auc_targeted:.4f}"
      f"   dAUC {observed:+.4f}")
print(f"  building null: {cfg.N_PERM} x {len(cfg.SEEDS)} = "
      f"{cfg.N_PERM * len(cfg.SEEDS)} model fits ...")

null = make_null(cfg.PERM_SEED, cfg.N_PERM)
summary = rs.summarize_null(null, observed, two_sided=True)

rng18 = np.random.default_rng(42)
single = evaluate_removal(rng18.choice(feats, size=k_drop, replace=False))

report.header("Figure 2b - targeted vs random 30% removal",
              f"N_PERM={cfg.N_PERM}, SEEDS={cfg.SEEDS}")
print(f"  observed dAUC        {summary['observed']:+.4f}")
print(f"  null mean / SD       {summary['null_mean']:+.4f} / {summary['null_sd']:.4f}")
print(f"  null 95% interval    [{summary['null_lo']:+.4f}, {summary['null_hi']:+.4f}]")
print(f"  perms >= observed    {summary['n_ge_observed']} / {summary['n_perm']}")
print(f"  z vs null            {summary['z']:+.3f}")
print(f"  {summary['p_formatted']}  (two-sided; floor {summary['p_floor']:.2g})")
print()
print(f"  single random draw (old cell 18)  dAUC {single:+.4f}"
      f"  -> percentile {100 * (null < single).mean():.0f} of this null")

print()
print(f"  stability: {cfg.N_PERM_SEEDS} permutation seeds x "
      f"{cfg.N_PERM_STABILITY} perms")
stab = rs.stability_sweep(make_null, observed, n_perm=cfg.N_PERM_STABILITY)
for r in stab:
    print(f"    seed {r['perm_seed_base']}  null [{r['null_lo']:+.4f}, "
          f"{r['null_hi']:+.4f}]  n_ge={r['n_ge_observed']}  "
          f"z={r['z']:+.2f}  p={r['p']:.4g}")
report.stability_note(stab)

report.save({"perm": np.arange(null.size), "null_diff": null}, "fig2b_null_draws")
report.save([{**summary, "single_draw_diff": single,
              "auc_baseline": auc_baseline, "auc_targeted": auc_targeted}],
            "fig2b_summary")
report.save(stab, "fig2b_stability")
