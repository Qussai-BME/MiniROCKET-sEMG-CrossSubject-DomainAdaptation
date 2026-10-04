# MiniROCKET sEMG — Cross-Subject Unsupervised Domain Adaptation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23136515.svg)](https://doi.org/10.5281/zenodo.23136515)

Code, per-subject results, tables and figures for the article
**"Cross-Subject sEMG Gesture Recognition with Reference MiniROCKET: A Multi-Seed Benchmark of Unsupervised Domain Adaptation on NinaPro DB2, DB3 and DB7"**
(manuscript under review).

**Authors:** Qussai Adlbi¹ ², Mohamad Ayham Darwich¹ ²
¹ Al-Andalus University for Medical Sciences, Syria · ² Pázmány Péter Catholic University, Budapest, Hungary
**Contact:** qussai.adlbi@au.edu.sy

> **Version 2.0.0 supersedes v1.0.0.** The CORAL implementation in v1.0.0 contained two errors, so the
> v1.0.0 conclusion that CORAL degrades accuracy does not hold and those results should not be used.
> See [CHANGELOG.md](CHANGELOG.md). The DOI above is the concept DOI and always resolves to the latest version.

## What is evaluated

- **Databases:** NinaPro DB2 (40 intact subjects, 49 movements), DB3 (11 transradial amputees, 17 movements), DB7 (20 intact + 2 amputee subjects, 40 movements). The rest class is excluded.
- **Features:** reference multivariate MiniROCKET (`sktime`), 10,000 kernels → 9,996 PPV features, 200-ms windows, 50 % overlap, fitted on the source windows of each fold.
- **Conditions:** Raw (inductive), CORAL, Transfer Component Analysis (TCA), Subspace Alignment (SA). An explicit-centering condition is run as a check only; it is numerically identical to Raw and is excluded from inference.
- **Protocol:** subject-held-out evaluation with a capped, deterministic source pool (≤ 5,000 windows per source subject, 50,000 in total; this is *not* full LOSO). The repetitions of the held-out subject are split, before windowing, into an unlabeled calibration half (the only target data the adaptation methods may use) and a disjoint final-test half on which all methods are scored.
- **Classifier:** ridge classifier on standardized features, regularization chosen by efficient leave-one-out CV on source labels only.
- **Seeds:** 42, 123, 2024, 7, 999 → 73 held-out subjects × 5 seeds; 1,825 method–fold evaluations including Centering.

## Main results

Accuracy (%), mean ± SD across subjects of seed-averaged accuracy (article Table 2).

| Database | Chance | Raw | CORAL | TCA | SA |
|---|---|---|---|---|---|
| DB7 (22 subjects, 40 classes) | 2.5 | 20.7 ± 7.1 | **22.0 ± 6.4** | 20.4 ± 6.4 | 7.7 ± 4.7 |
| DB3 (11 subjects, 17 classes) | 5.9 | 8.9 ± 2.9 | 8.4 ± 2.7 | 9.7 ± 3.1 | 6.1 ± 0.6 |
| DB2 (40 subjects, 49 classes) | 2.0 | 11.5 ± 3.3 | **12.5 ± 2.3** | 11.2 ± 2.5 | 2.9 ± 0.7 |

- Corrected CORAL reduces the source–target covariance distance by 81–90 % in all 365 fold–seed evaluations, but improves accuracy by only about 1 percentage point at its default regularization (DB7 +1.36 pp, adjusted p = 0.041; DB2 +0.99 pp, p = 0.011; DB3 not detectable). Gains are larger for macro-F1 (+2.5 / +2.0 / +1.8 pp on DB7 / DB3 / DB2).
- TCA gives no significant accuracy gain; SA lowers accuracy on every database.
- A descriptive single-seed DB7 sweep shows further CORAL gains at stronger regularization (24.3 % at λ = 0.1), so the default does not exhaust its potential.

All numbers above are regenerated from the per-subject result files by the scripts in `paper_reproduction/`.

## Repository layout

```
config.py, data_loader.py, feature_extractors.py, minirocket.py,
domain_adaptation.py, subject_split.py, optimized_pipeline.py, statistical_utils.py   # library code

p08_run_multiseed_sweep.py           # main experiment: all conditions x databases x seeds (uses optimized_pipeline)
p10_run_sensitivity_analysis.py      # CORAL lambda / TCA, SA dimension sweep (DB7, seed 42)

paper_reproduction/                  # regenerates every table and figure of the article from the shipped results
    run_all.py                       #   runs the two scripts below
    p13_article_tables_and_figures.py
    make_figure5_and_table_s13.py    #   Figure 5 and Table S13 (synthetic CORAL verification)

p01a_run_main_loso_benchmark.py      # single-seed runner; provides apply_adaptation() used by p10
p01b, p02, p03a-b, p04a-c, p05, p06, p06b, p07   # additional analysis scripts (see note below)

tools/                               # synthetic correctness tests (incl. audit_coral_synthetic_figure5.py) and data diagnostics
dev_history/                         # archived one-time patch scripts
outputs/results/                     # results_<db>_<method>__canonical__seed<S>__shared.json  (75 files),
                                     #   MULTISEED_SUMMARY_*.json, sensitivity_analysis_db7_feature_reuse.json
outputs/tables/, outputs/figures/    # article tables (CSV) and figures (PNG + PDF)
```

Note: the scripts `p01b`, `p02`, `p03`–`p07` belong to the analysis pipeline of v1.0.0 and are kept for
completeness. **The tables and figures of the article are produced by the scripts in `paper_reproduction/` only**; the outputs of the other
scripts are not part of this release.

### Output files and article items

| File(s) in `outputs/tables/` | Article item |
|---|---|
| `Table1_data_volumes` | Table 1 |
| `Table2_main_results` | Table 2 |
| `Table3_friedman` | Table 3 (accuracy; balanced accuracy and macro-F1 in the same file) |
| `Table4_effects_vs_raw` | Table 4 |
| `Table5_coral_covariance_shift` | Table 5 |
| `Table6_population_subgroups` | Table 6 |
| `Table7_sensitivity_db7_seed42` | Table 7 |
| `TableS_pairwise_contrasts_all_metrics`, `TableS_restricted_family_three_contrasts` | full pairwise tests; three-contrast family |
| `TableS_per_subject_metrics`, `TableS_per_class_f1`, `TableS_seed_stability`, `TableS_timing` | per-subject, per-class, seed and timing supplements |
| `TableS_prediction_balance`, `TableS_gain_correlations` | Section 3.5 and Section 3.4 statistics |
| `TableS13_synthetic_coral` | Supplementary Table S13 (synthetic verification of CORAL) |

Figures `figure2`–`figure12` and `figureS1` in `outputs/figures/` carry the numbers of the corresponding article figures. Figure 1 (study design) is a drawing and is not data-derived.

## Installation and reproduction

```bash
git clone https://github.com/Qussai-BME/MiniROCKET-sEMG-CrossSubject-DomainAdaptation.git
cd MiniROCKET-sEMG-CrossSubject-DomainAdaptation
pip install -r requirements.txt            # Python >= 3.10
```

**Regenerate every table and figure from the shipped results (no NinaPro data needed):**

```bash
python paper_reproduction/run_all.py
```

**Correctness checks (synthetic data, a few minutes):**

```bash
python tools/audit_coral_synthetic_test.py
python tools/audit_coral_synthetic_figure5.py      # the five synthetic problems of Figure 5 / Table S13
python tools/audit_calibration_split_synthetic_test.py
python tools/audit_dataloader_synthetic_test.py
```

**Re-run the experiments** (requires the NinaPro download; set `DB_PATHS` in `config.py` first):

```bash
python p08_run_multiseed_sweep.py --db db7 --extractor canonical --seeds 42,123,2024,7,999 --num_kernels 10000 --output_dir outputs/results
python p10_run_sensitivity_analysis.py --db db7 --extractor canonical --resume
python paper_reproduction/run_all.py
```

A full run is computationally heavy (feature extraction dominates: roughly 6–7 minutes per fold on the hardware used).

## Data availability

NinaPro DB2, DB3 and DB7 are publicly available at <https://ninapro.hevs.ch>. The raw recordings are not redistributed here.

## Not included in this release

- Resume checkpoints, feature caches and run logs (regenerable and large).
- A within-subject reference, window-size/sample-size ablations, kernel-importance and eigenvalue diagnostics: not reported in the article, and the earlier outputs were produced with the v1.0.0 CORAL implementation.

## Citation

Please cite the article once published and the archived software release; see [CITATION.cff](CITATION.cff).
Update this section with the final journal reference after acceptance.

## AI-assistance statement

A large language model (Anthropic Claude) was used for code debugging and review and for writing statistical-analysis and figure scripts. All analyses were specified, supervised and verified by the authors, who take full responsibility for the code and results (see the article's declaration).

## License

MIT — see [LICENSE](LICENSE).
