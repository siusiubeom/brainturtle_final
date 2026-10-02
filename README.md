# Brainturtle

*Chronological age matches acoustic models of Alzheimer's speech and deconfounding recovers signal complementary to age*
(Siu Beom, Chae Young Kim, Isaac Jinwon Yi & Sun Shin Yi; Konkuk University, Emory University).

---

## Data

| Dataset | Role | Access |
|---|---|---|
| DementiaBank Pitt Corpus (Cookie Theft) | training and internal evaluation | https://talkbank.org/dementia/access/English/Pitt.html (requires approval) |
| DementiaBank Korean Kang Corpus | external validation | https://talkbank.org/dementia/access/Korean/Kang.html (requires approval) |
| ADReSSo 2021 | external validation | https://media.talkbank.org/dementia/English/0extra/ADReSSo (requires approval) |
| Mozilla Common Voice (en-AU, v24) | normative age reference | https://datacollective.mozillafoundation.org/datasets/cmko7havo02f5nw07rbwwhowe (public) |

This repository holds the code, the figures and the speaker-level acoustic feature tables with
the feature rankings (`data/`); the Zenodo archive holds the code and the figures. Raw audio (~6.9 GB), the corpus demographic
files (`data/PItt-data.xlsx`, `demo-kang.xlsx`, Common Voice metadata), saved classifiers and
result tables are not included; the corpora are available from the sources listed above.

### Expected directory layout for the audio

```
data/
├── pitt_dementia_cookie/          Pitt, AD recordings (.wav)
├── pitt_control_cookie/           Pitt, control recordings (.wav)
├── korean/
│   ├── ad/                        Kang, MCI recordings (.wav)
│   ├── cn/                        Kang, healthy-control recordings (.wav)
│   └── script/{MCI,HC}/           Kang CHAT transcripts (.cha), used to cut investigator turns
├── addresso2021/
│   ├── ad/
│   └── cn/
├── audio_files/                   Common Voice clips
└── commonvoice-v24_en-AU.csv      Common Voice metadata
```

---

## Parameters

Resampling, seeds and thresholds are defined once in `brainturtle/config.py`; Supplementary
Table 1 is rendered from it by `analyses/h_supp_table1.py`, which also carries the fixed
signal-processing values below.

| Parameter | Value |
|---|---|
| Sample rate (`SR`) | 16,000 Hz, mono |
| RMS normalization target | −20 dB |
| Silence threshold (`top_db`) | 30 dB |
| MFCC coefficients | 13 |
| Summary statistics per stream | mean, SD, skewness, kurtosis, p25, p75 |
| Total acoustic features | 110 (78 MFCC, 18 spectral, 6 pitch, 6 energy, 2 voice quality: shimmer, HNR) |
| Classifier | LightGBM, `class_weight="balanced"` |
| Cross-validation folds | 5 speaker-level stratified 80 / 20 splits, seeds (`SEEDS`) 0, 1, 2, 3, 4 |
| Age-sensitivity ranking | LightGBM age regressor on Common Voice, 600 trees, learning rate 0.03, interventional SHAP |
| Removal grid, pre-specified | 0, 10, 20, 30, 40 % of the SHAP ranking |
| Removal grid, exploratory | 0–40 % in 5 % steps |
| Primary removal threshold | 30 % (33 of 110 features) |
| Age strata | 50–60, 60–70, 70–80 years |
| Bootstrap resamples (`NBOOT`) | 10,000 |
| Permutation iterations (`NPERM`) | 10,000 (p-value floor 2/10,001) |
| Permutation stability replicates | 5 seeds × 1,000 permutations |
| Bootstrap / permutation seed | 42 |
| Incremental-value logistic model | 5-fold stratified CV × 20 repeats, out-of-fold AUC |
| Significance | α = 0.05, percentile 95 % CI (2.5–97.5) |

---

## Layout

```
brainturtle/        the package: paths, constants, loaders, evaluation, resampling
analyses/           one script per result, each writing CSVs to results/n10000/
data/               feature tables and feature rankings read by the analyses
figures/            one script per figure (figure6_pipeline.tex is TikZ) and the rendered figures
run_all.py          runs the analyses in dependency order
```

---

## Reproduce

```bash
pip install -e .                           # pinned versions from pyproject.toml
pip install -e ".[geometry]"               # umap-learn + hdbscan, Supplementary Figure 9 only

python analyses/z_consistency_checks.py    # provenance guards, run first
python run_all.py --cheap                  # ~25 min, no retraining
python run_all.py                          # ~12 h at NPERM=10000
python figures/sources.py                  # every panel's source table exists

for f in figures/figure*.py figures/supp_fig*.py; do python "$f"; done
python figures/figure6_assets.py && (cd figures && pdflatex figure6_pipeline.tex) && python figures/figure6_render.py
python figures/package_figures.py          # figures.zip with manuscript names
```

The permutation nulls and the Isolation Forest cells are estimator-dependent: install the
pinned scikit-learn before comparing a rerun with the manuscript.
