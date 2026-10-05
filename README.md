# MiniROCKET sEMG — Cross-Subject Benchmark of Unsupervised Domain Adaptation (NinaPro DB2, DB3, DB7)

[![DOI (this version, v2.0.0)](https://zenodo.org/badge/DOI/10.5281/zenodo.23136515.svg)](https://doi.org/10.5281/zenodo.23136515)
[![DOI (all versions)](https://zenodo.org/badge/DOI/10.5281/zenodo.21940500.svg)](https://doi.org/10.5281/zenodo.21940500)

Companion code and results of the manuscript
**"Cross-Subject sEMG Gesture Recognition with Reference MiniROCKET: A Multi-Seed Benchmark of Unsupervised Domain Adaptation on NinaPro DB2, DB3 and DB7"**
(Q. Adlbi, M. A. Darwich). **Cite this version:** https://doi.org/10.5281/zenodo.23136515 (the citation of the article will be added once published).

> **Version note.** v2.0.0 replaces v1.0.0 (https://zenodo.org/records/21940501, *superseded*). v1.0.0 used a hand-written PPV extractor,
> a CORAL function with two errors (an extra re-colouring of the target and a transposition error in the transform), a window-level
> within-subject baseline and a single seed. Its numerical results and conclusions are **not** comparable with, and should not be used instead of, those of v2.0.0.

## What this study does
* Features: the reference **multivariate MiniROCKET** transform (sktime `MiniRocketMultivariate`, 10,000 kernels → 9,996 PPV features) + ridge classifier.
* Evaluation: **inductive subject-held-out evaluation with a capped source pool ("capped-pool LOSO")** — *not* full LOSO. Each fold uses the first
  10–11 source subjects in subject-ID order until a 50,000-window cap is reached (DB7: 11 of 21, DB2: 11 of 39, DB3: 10 of 10); 200-ms windows, 50 % overlap.
* UDA conditions (**CORAL, TCA, SA**) use *unlabeled* calibration repetitions of the held-out subject; calibration and final-test repetitions are disjoint
  (split by repetition **before** windowing). Target labels are used only offline to build the class-stratified evaluation split.
* 73 held-out subjects × 5 conditions × 5 seeds (42, 123, 2024, 7, 999) = 1,825 method–fold evaluations. Statistics: subject-level Friedman, paired two-sided Wilcoxon with Holm–Šidák correction, bootstrap CIs.

## Main results (mean accuracy over held-out subjects, %; chance = 1/K)
| Database | Chance | Raw | CORAL | TCA | SA |
|---|---|---|---|---|---|
| DB7 (40 classes) | 2.5 | 20.7 | 22.0 | 20.4 | 7.7 |
| DB3 (17 classes) | 5.9 | 8.9 | 8.4 | 9.7 | 6.1 |
| DB2 (49 classes) | 2.0 | 11.5 | 12.5 | 11.2 | 2.9 |

At its pre-specified default regularization (λ = 10⁻³), CORAL reduced source–target covariance distance in all 365 fold–seed evaluations (mean reduction 81–90 %) but raised accuracy by only
+1.36 pp (DB7) and +0.99 pp (DB2), with larger macro-F1 gains; TCA gave no significant accuracy gain; SA reduced accuracy. A descriptive single-seed DB7 sweep did not identify a regularization optimum.
Full tables, per-subject results, seeds, sensitivity and timing are in the article and its Supplementary Material.

## Repository guide
* `config.py`, `data_loader.py`, `subject_split.py`, `feature_extractors.py`, `domain_adaptation.py`, `statistical_utils.py`, `optimized_pipeline.py` — core modules.
* `p01a_run_main_loso_benchmark.py`, `p08_run_multiseed_sweep.py` — main experiment and multi-seed driver; `p10b_run_sensitivity_feature_reuse.py` — DB7 sensitivity sweep.
* `p01b_run_within_subject_baseline.py`, `p12_close_paper_gaps.py` — additional experiments **not reported** in the article.
* `tools/audit_coral_synthetic_figure5.py` — synthetic Gaussian verification of CORAL (Figure 5, Supplementary Table S13). Run: `python tools/audit_coral_synthetic_figure5.py` → expects `PASS` and a 78.5–98.7 % reduction range.
* `paper_reproduction/` — regenerates every number, table and figure of the article from the result files (see its `README.md`).
* NinaPro data are **not** included; download them from https://ninapro.hevs.ch and set `NINAPRO_DB2`, `NINAPRO_DB3`, `NINAPRO_DB7`.

## Scope and limitations (see the article)
Capped, deterministic source pool; class-stratified calibration/test split built with target labels offline (an evaluation convenience, not a deployment protocol);
CORAL evaluated at its default regularization; offline classification only — no clinical, real-time or embedded claims.

## License
MIT (see `LICENSE`).
