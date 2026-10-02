"""Gender removal as a negative control."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from brainturtle import config as cfg, data, evaluate, featuresets as fs
from brainturtle import report, resampling as rs

report.run_banner("g_gender_control")

feats = fs.common_features()
k_drop = int(len(feats) * cfg.PRIMARY_THRESHOLD)
auc_baseline = evaluate.pooled_auc(feats)

gender_rank = pd.read_csv(cfg.DATA / "gender_shap_ranking.csv")
if "mean_abs_shap" in gender_rank.columns:
    gender_rank = gender_rank.sort_values("mean_abs_shap", ascending=False)
ranked = [f for f in gender_rank["feature"] if f in set(feats)]
drop_gender = ranked[:k_drop]
keep_gender = [f for f in feats if f not in set(drop_gender)]

auc_gender = evaluate.pooled_auc(keep_gender)
observed = auc_gender - auc_baseline


def evaluate_removal(drop_feats):
    keep = [f for f in feats if f not in set(drop_feats)]
    return evaluate.pooled_auc(keep) - auc_baseline


def make_null(seed, n_perm):
    return rs.subset_permutation_null(evaluate_removal, feats, k_drop,
                                      n_perm=n_perm, seed=seed)


print(f"  baseline AUC {auc_baseline:.4f} -> gender-targeted "
      f"{auc_gender:.4f}   dAUC {observed:+.4f}")
print(f"  building null: {cfg.N_PERM} random {cfg.PRIMARY_THRESHOLD:.0%} "
      f"removals ...")
null = make_null(cfg.PERM_SEED, cfg.N_PERM)
summary = rs.summarize_null(null, observed, two_sided=True)

report.header("Supplementary Figure 9 - gender removal vs random null",
              f"N_PERM={cfg.N_PERM}, SEEDS={cfg.SEEDS}")
print(f"  observed dAUC        {summary['observed']:+.4f}")
print(f"  null mean / SD       {summary['null_mean']:+.4f} / {summary['null_sd']:.4f}")
print(f"  null 95% interval    [{summary['null_lo']:+.4f}, {summary['null_hi']:+.4f}]")
print(f"  perms >= observed    {summary['n_ge_observed']} / {summary['n_perm']}")
print(f"  z vs null            {summary['z']:+.3f}")
print(f"  {summary['p_formatted']}  (two-sided)")
print(f"  percentile of observed within null: "
      f"{100 * (null < observed).mean():.0f}")

inside = summary["null_lo"] <= observed <= summary["null_hi"]
print()
if inside:
    print("  VERDICT: the gender-targeted removal falls INSIDE the random-removal")
    print("  null. The decrease is what removing an arbitrary set of this size")
    print("  does. It does not show that gender features carry disease signal,")
    print("  and it cannot serve as a specificity control in its current form.")
else:
    print("  VERDICT: the gender-targeted removal falls outside the null.")

stab = rs.stability_sweep(make_null, observed, n_perm=cfg.N_PERM_STABILITY)
report.stability_note(stab)

orphan = cfg.DATA / "gender_sensitive_features_to_drop.csv"
if orphan.exists():
    listed = pd.read_csv(orphan, header=None)[0].astype(str).tolist()
    same = set(listed) == set(drop_gender)
    print()
    print(f"  gender_sensitive_features_to_drop.csv: {len(listed)} features, "
          f"{'matches' if same else 'DIFFERS from'} the pure-SHAP "
          f"{cfg.PRIMARY_THRESHOLD:.0%} set ({len(drop_gender)})")
    if not same:
        print("    -> it was built by a different rule; do not compare it "
              "against the age figures without regenerating.")

report.save({"perm": np.arange(null.size), "null_diff": null},
            "suppfig9_null_draws")
report.save([{**summary, "auc_baseline": auc_baseline,
              "auc_gender_targeted": auc_gender,
              "observed_inside_null_95": inside}], "suppfig9_summary")
report.save(stab, "suppfig9_stability")
report.save(pd.DataFrame({"feature": drop_gender}), "drop30_gender")
