# Figure set — regeneration and manuscript mapping

Every figure is written by a `.py` script in this directory and saved as
**TIFF (600 dpi, LZW), PNG (300 dpi) and JPG (300 dpi)** by `figstyle.save()`.

Regenerate the whole set from the repo root:

```bash
for f in figures/figure*.py figures/supp_fig*.py; do python "$f"; done
```

Figure 6 is the exception: it is a TikZ/pgfplots drawing. Every count in it
is a macro and every bar a data table, both written by `figure6_assets.py`
into `figures/assets/`, so the drawing is compiled from the analysis rather
than typed. `figure6_panels.tex` holds the preamble, palette and the bar and
histogram helpers; `figure6_pipeline.tex` is the drawing. The `.tex` must be
compiled from inside `figures/` because it reads `assets/` by relative path.

```bash
python figures/figure6_assets.py                       # assets/fig6_*.tex|.dat
(cd figures && pdflatex figure6_pipeline.tex)          # -> figure6_pipeline.pdf
python figures/figure6_render.py                       # PDF -> png/jpg/tiff
```

Every script reads from `results/n10000/`, so a panel cannot silently come
from a different run size than the number quoted beside it in the text. Run
the analyses first if that directory is empty (`python run_all.py`).

The script filenames predate the current supplementary numbering and do not
match it. Use the table below rather than the filename.

## Main figures

| Manuscript | Script | Output basename | Data source |
|---|---|---|---|
| Figure 1 | `figure1_age_benchmark.py` | `figure1_age_benchmark` | `fig1_age_benchmark.csv`, `fig1_age_distribution.csv`, `fig1_age_roc.csv`, `kang_age_benchmark.csv`, `kang_speaker_table.csv` |
| Figure 2 | `figure2_removal.py` | `figure2_removal` | `fig2a_removal_ladder.csv`, `fig2b_null_draws.csv`, `fig2b_summary.csv`, `fig2c_agebins.csv` |
| Figure 3 | `figure3_category.py` | `figure3_category` | `fig3_category_rates.csv`, `fig3_mfcc_subfamily.csv` |
| Figure 4 | `figure4_incremental.py` | `figure4_incremental` | `fig4_incremental.csv`, `fig4_kang_incremental.csv` |
| Figure 5 | `figure4_external.py` | `figure4_external` | `fig4_external.csv`, `kang_age_benchmark.csv`, `kang_speaker_table.csv`, `kang_threshold_sweep.csv` |
| Figure 6 | `figure6_pipeline.tex` + `figure6_assets.py` + `figure6_render.py` | `figure6_pipeline` | TikZ schematic in the flow-figure layout: a) corpus, exclusion (dashed), modelled set with family bar, evaluation; b) extraction diagram; c) ranking from `age_shap_ranking.csv`; counts via `assets/fig6_numbers.tex` |

## Supplementary figures

| Manuscript | Script | Output basename | Data source |
|---|---|---|---|
| S1 | `supp_fig8_pitt_demographics.py` | `supplementary_fig8_pitt_demographics` | `data.pitt_with_age()`, `data.commonvoice()` |
| S2 | `supp_fig13_age_ranking_stability.py` | `supplementary_fig13_age_ranking_stability` | `age_model_partition.csv`, `age_model_ranking_check.csv`, `age_model_shap_values.csv` |
| S3 | `figure1_suppression.py` | `figure1_suppression` | `age_model_shap_values.csv`, `fig1_suppression.csv` |
| S4 | `supp_fig14_detector_robustness.py` | `supplementary_fig14_detector_robustness` | `item8_detector_robustness.csv` |
| S5 | `supp_fig1_removal_ladder.py` | `supplementary_fig1_removal_ladder` | `fig2a_removal_ladder.csv` |
| S6 | `supp_fig9_gender_control.py` | `supplementary_fig9_gender_control` | `suppfig9_null_draws.csv`, `suppfig9_summary.csv` |
| S7 | `supp_fig7_spearman_heatmap.py` | `supplementary_fig7_spearman_heatmap` | computed from Common Voice |
| S8 | `supp_fig2_shap_spearman_overlap.py` | `supplementary_fig2_shap_spearman_overlap` | computed from Common Voice + `age_shap_ranking.csv` |
| S9 | `supp_fig3_geometry.py` | `supplementary_fig3_geometry` | `geometry_embedding.csv`, `geometry_silhouettes.csv`, `geometry_summary.csv` |
| S10 | `supp_fig5_calibration.py` | `supplementary_fig5_calibration` | `calibration.csv`, `calibration_bins.csv`, `ece_conventions.csv` |
| S11 | `supp_fig11_recording_duration.py` | `supplementary_fig11_recording_duration` | `recording_stats.csv`, `recording_stats_summary.csv` |
| S12 | `supp_fig12_speech_ratio.py` | `supplementary_fig12_speech_ratio` | `recording_stats.csv`, `recording_stats_summary.csv` |

`supp_fig4_kang_conditions.py` is not in the manuscript. Its values are
reported as Supplementary Table 2, emitted by
`analyses/l_supplementary_audit.py`.
