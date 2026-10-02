"""Run the analysis set in dependency order."""
import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from brainturtle import config as cfg

ROOT = Path(__file__).resolve().parent

PLAN = [
    ("analyses/z_consistency_checks.py", "cheap", 0.0,
     "provenance guards; run first"),
    ("analyses/h_supp_table1.py", "cheap", 0.0,
     "Supp Table 1 + Table 1, rendered from config"),
    ("analyses/e_category.py", "cheap", 0.0,
     "Figure 3 category rates (item 9)"),
    ("analyses/b_removal_sweep.py", "cheap", 0.0,
     "Figure 2a + Supp S1 ladder; bootstrap only"),
    ("analyses/d_agebins.py", "cheap", 0.0,
     "Figure 2c; label permutation"),
    ("analyses/i_calibration.py", "cheap", 0.0,
     "Calibration and operating point"),
    ("analyses/f_external.py", "cheap", 0.0,
     "Figure 4 external validation; label permutation"),
    ("analyses/c_removal_null.py", "retrain", 0.41,
     "Figure 2b targeted-vs-random null"),
    ("analyses/g_gender_control.py", "retrain", 0.41,
     "Gender negative control"),
    ("analyses/a_suppression.py", "retrain", 1.04,
     "Supp S15b/c suppression; 11 feature counts"),
    ("analyses/o_curves.py", "cheap", 0.0,
     "ROC coordinates, reliability bins, ECE under every binning convention"),
    ("analyses/l_supplementary_audit.py", "cheap", 0.0,
     "supplementary claim audit + Kang threshold sweep"),
    ("analyses/j_ad_drivers.py", "cheap", 0.0,
     "post-cut AD SHAP drivers; inverted-U mechanism"),
    ("analyses/n_geometry.py", "cheap", 0.0,
     "Supp S3 cluster geometry, both feature spaces"),
    ("analyses/p_recording_stats.py", "cheap", 0.0,
     "Supp S11/S12 duration and speech-ratio tests"),
    ("analyses/q_age_model_shap.py", "cheap", 0.0,
     "Supp S13/S15 age-model SHAP; reproduces the canonical ranking"),
    ("analyses/r_age_benchmark.py", "cheap", 0.0,
     "Figures 1 and 4: age benchmark and incremental value"),
    ("analyses/k_detector_robustness.py", "retrain", 1.04,
     "Supp S14 suppression under four detectors x two rankings"),
    ("analyses/s_kang_age_benchmark.py", "cheap", 0.0,
     "Figure 1 lower row: the age benchmark on the Kang corpus"),
    ("analyses/t_reviewer_checks.py", "cheap", 0.0,
     "voice quality extraction audit, feature-age correlation, "
     "converter sensitivity"),
]


"""Measured per-iteration costs on 16 cores.

`cheap` scripts do not retrain, but they are not free at large N: a single
roc_auc_score on ~300 speakers is ~1 ms, and b_removal_sweep does 9 bootstrap
CIs plus 8 paired difference tests while d_agebins does 20 permutation nulls
plus 20 CIs. At N=10,000 that is several hundred thousand AUC evaluations.
The multiplier below is how many N-sized resampling loops each script runs.
"""
CHEAP_LOOPS = {
    "analyses/z_consistency_checks.py": 0,
    "analyses/h_supp_table1.py": 0,
    "analyses/e_category.py": 0,
    "analyses/b_removal_sweep.py": 25,
    "analyses/d_agebins.py": 40,
    "analyses/i_calibration.py": 0,
    "analyses/f_external.py": 12,
    "analyses/o_curves.py": 0,
    "analyses/l_supplementary_audit.py": 6,
    "analyses/j_ad_drivers.py": 0,
    "analyses/n_geometry.py": 0,
    "analyses/p_recording_stats.py": 0,
    "analyses/q_age_model_shap.py": 0,
    "analyses/r_age_benchmark.py": 6,
}
SEC_PER_AUC = 0.001


def estimate(script, cls, per_perm):
    if cls == "cheap":
        loops = CHEAP_LOOPS.get(script, 0)
        return 20.0 + loops * cfg.N_PERM * SEC_PER_AUC
    stability = cfg.N_PERM_SEEDS * cfg.N_PERM_STABILITY
    return per_perm * (cfg.N_PERM + stability)


def show_plan(only=None):
    print(f"N_PERM={cfg.N_PERM}  N_BOOT={cfg.N_BOOT}  "
          f"stability={cfg.N_PERM_SEEDS}x{cfg.N_PERM_STABILITY}\n")
    total = 0.0
    for script, cls, per, note in PLAN:
        if only and cls != only:
            continue
        secs = estimate(script, cls, per)
        total += secs
        print(f"  {script:38s} {cls:8s} ~{secs/60:6.1f} min   {note}")
    print(f"\n  estimated total: {total/60:.0f} min ({total/3600:.1f} h)")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cheap", action="store_true",
                    help="skip analyses that retrain models")
    ap.add_argument("--list", action="store_true", help="show plan and exit")
    args = ap.parse_args()

    only = "cheap" if args.cheap else None
    if args.list:
        show_plan(only)
        return 0

    show_plan(only)
    print()
    failed = []
    for script, cls, _, _ in PLAN:
        if only and cls != only:
            continue
        print(f"\n{'=' * 78}\n>>> {script}\n{'=' * 78}")
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT)
        dt = time.perf_counter() - t0
        status = "ok" if r.returncode == 0 else f"EXIT {r.returncode}"
        print(f"<<< {script}  {status}  {dt/60:.1f} min")
        if r.returncode != 0:
            failed.append(script)

    print(f"\n{'=' * 78}")
    if failed:
        print(f"{len(failed)} script(s) reported a problem: {', '.join(failed)}")
        print("z_consistency_checks.py exits nonzero by design when a "
              "provenance guard trips -- read its output before rebuilding "
              "figures.")
    else:
        print("all analyses completed")
    print(f"results -> {cfg.RESULTS / f'n{cfg.N_PERM}'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
