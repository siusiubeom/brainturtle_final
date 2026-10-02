"""Single source of truth for every resampling constant in the paper."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("PROJECT_ROOT",
                           Path(__file__).resolve().parents[1]))
DATA = ROOT / "data"
FEATURES = DATA / "features"
MODEL_DIR = ROOT / "models_pitt_full"
FEAT_DIR = ROOT / "features_pitt_full"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def _int_env(name, default):
    return int(os.environ.get(name, default))


for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

MODEL_THREADS = 1


N_PERM = _int_env("NPERM", 10_000)
N_BOOT = _int_env("NBOOT", N_PERM)

N_PERM_SEEDS = _int_env("NPERMSEEDS", 5)
PERM_SEED_BASES = list(range(N_PERM_SEEDS))

N_PERM_STABILITY = _int_env("NPERMSTAB", 1_000)

BOOT_SEED = 42
PERM_SEED = 42
MODEL_SEED_BASE = 42

SEEDS = [0, 1, 2, 3, 4]
TEST_SIZE = 0.2

REMOVAL_GRID_PRESPECIFIED = [0.0, 0.10, 0.20, 0.30, 0.40]
REMOVAL_GRID_EXPLORATORY = [0.0, 0.05, 0.10, 0.15, 0.20,
                            0.25, 0.30, 0.35, 0.40]
PRIMARY_THRESHOLD = 0.30

KEEP_PERCENTS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.90, 1.0]

AGE_BINS = [(50, 60), (60, 70), (70, 80), (0, 100)]

CANONICAL_RULE = "shap_pct"
LEGACY_UNION_RHO = 0.10
LEGACY_UNION_SHAP_PCT = 0.20

EXTERNAL_COHORTS = {
    "ADReSSo-2021": {
        "path": FEATURES / "addresso2021_speaker_features.csv",
        "role": "primary",
    },
    "Kang": {
        "path": DATA / "korean_speaker_features.csv",
        "role": "primary",
    },
    "Kang (investigator-removed)": {
        "path": DATA / "korean_speaker_features_cut.csv",
        "role": "sensitivity",
    },
}

ALPHA = 0.05
CI_LOW, CI_HIGH = 2.5, 97.5


def p_floor(n_perm=None, two_sided=True):
    """Smallest p an add-one permutation test can produce."""
    n = N_PERM if n_perm is None else n_perm
    return (2 if two_sided else 1) / (n + 1)
