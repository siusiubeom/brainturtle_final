"""Supplementary Table 1 and Table 1, rendered from brainturtle.config."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from brainturtle import config as cfg, data, featuresets as fs, report

report.run_banner("h_supp_table1")

preprocessing = [
    ("Sample rate (SR)", "16,000 Hz"),
    ("RMS normalization target", "-20 dB"),
    ("Silence threshold (top_db)", "30 dB"),
    ("MFCC coefficients", "13"),
    ("Total acoustic features", str(len(fs.common_features()))),
    ("Cross-validation folds", f"{len(cfg.SEEDS)} (speaker-level stratified)"),
    ("Training seeds", ", ".join(str(s) for s in cfg.SEEDS)),
    ("Held-out fraction per split", f"{cfg.TEST_SIZE:.0%}"),
]

resampling = [
    ("Bootstrap resamples (AUC confidence intervals)", f"{cfg.N_BOOT:,}"),
    ("Bootstrap resamples (AUC difference test)", f"{cfg.N_BOOT:,}"),
    ("Bootstrap resamples (external validation, speaker-level)",
     f"{cfg.N_BOOT:,}"),
    ("Permutation iterations (random-removal null)", f"{cfg.N_PERM:,}"),
    ("Permutation iterations (label permutation vs chance)", f"{cfg.N_PERM:,}"),
    ("Permutation RNG seeds (stability check)", str(cfg.N_PERM_SEEDS)),
    ("Permutations per stability replicate", f"{cfg.N_PERM_STABILITY:,}"),
    ("Bootstrap seed", str(cfg.BOOT_SEED)),
    ("Permutation seed", str(cfg.PERM_SEED)),
    ("Smallest reportable p (two-sided)", f"{cfg.p_floor(two_sided=True):.2g}"),
    ("Smallest reportable p (one-sided)", f"{cfg.p_floor(two_sided=False):.2g}"),
]

grids = [
    ("Removal grid, pre-specified",
     ", ".join(f"{p:.0%}" for p in cfg.REMOVAL_GRID_PRESPECIFIED)),
    ("Removal grid, exploratory",
     ", ".join(f"{p:.0%}" for p in cfg.REMOVAL_GRID_EXPLORATORY)),
    ("Primary threshold carried to external validation",
     f"{cfg.PRIMARY_THRESHOLD:.0%}"),
    ("Age-sensitivity ranking rule",
     f"top {cfg.PRIMARY_THRESHOLD:.0%} by mean |SHAP|, no correlation union"),
]

supp = pd.DataFrame(
    [{"section": "Preprocessing and features", "parameter": k, "value": v}
     for k, v in preprocessing]
    + [{"section": "Analysis grids", "parameter": k, "value": v}
       for k, v in grids]
    + [{"section": "Resampling", "parameter": k, "value": v}
       for k, v in resampling])

report.header("Supplementary Table 1 (generated from config)")
for section in supp["section"].unique():
    print(f"\n  {section}")
    for r in supp[supp.section == section].itertuples():
        print(f"    {r.parameter:58s} {r.value}")

report.save(supp, "supp_table1_parameters")
report.save(supp[supp.section == "Resampling"], "methods_parameters")

t1 = data.describe_cohorts()
report.header("Table 1 - cohort characteristics",
              "unavailable cells carry a stated reason, never a blank")
report.show(t1, ["cohort", "n_speakers", "n_ad", "n_control",
                 "age_mean", "age_sd"])
print()
for r in t1.itertuples():
    if r.age_note or r.sex_note:
        print(f"    {r.cohort:30s} age: {r.age_note or 'available':40s} "
              f"sex: {r.sex_note or 'available'}")

pitt = data.pitt()
bal = data.class_balance(pitt["label"].values)
print()
print(f"  Pitt base rate {bal['base_rate']:.4f}; an uninformative predictor "
      f"scores Brier {bal['brier_uninformative']:.4f}")
print("  (not 0.25 -- that assumes perfect balance. Any Brier skill score "
      "must use this reference.)")

report.save(t1, "table1_cohorts")
report.save([bal], "brier_reference")
