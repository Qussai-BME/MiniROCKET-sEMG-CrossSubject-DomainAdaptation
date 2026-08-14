# Why MiniROCKET PPV Features Resist Domain Adaptation

**A Mechanistic Study of Cross-Subject sEMG Gesture Recognition Failure**

Qussai Adlbi¹ ², Mohamad Ayham Darwich¹ ²
¹ Al-Andalus University for Medical Sciences, Syria · ² Pázmány Péter Catholic University, Budapest, Hungary
Correspondence: qussai.adlbi@au.edu.sy, a.darwich@au.edu.sy

*Manuscript under review. This repository will be linked from the published paper once it is accepted; a citation entry is provided below and will be updated with the final DOI.*

## Overview

Cross-subject generalization is a central obstacle to clinical deployment of surface EMG (sEMG) gesture recognition. MiniROCKET — a fast random-convolutional-kernel time-series classifier — achieves strong *within-subject* accuracy on sEMG, but its behavior under strict Leave-One-Subject-Out (LOSO) cross-validation, and its compatibility with standard unsupervised domain adaptation (DA), had not been systematically characterized. This repository contains the full pipeline for that evaluation: MiniROCKET with five feature-space treatments (no adaptation, feature centering, CORAL, Transfer Component Analysis (TCA), and Subspace Alignment (SA)) under strict LOSO cross-validation on three NinaPro databases — DB2 (intact-limb, N=40), DB3 (transradial/transhumeral amputee, N=11), and DB7 (mixed, N=22) — totaling 15 experimental conditions and 365 subject-level LOSO folds.

**This is a negative-result paper.** None of the four domain adaptation methods tested improved on the unadapted baseline; CORAL made things significantly *worse*. The repository also contains the mechanistic follow-up analysis (covariance-eigenvalue diagnostics, domain-shift metrics, kernel-importance analysis) that traces this failure to a structural property of MiniROCKET's Proportion-of-Positive-Values (PPV) feature space.

## Key Results

| Database | Population | LOSO accuracy (raw) | Chance | Ratio× | Within-subject accuracy | Within/LOSO ratio× |
|---|---|---|---|---|---|---|
| DB7 | Mixed (22 subjects, 40 classes) | 9.15 ± 2.21% | 2.51% | 3.66× | 42.50 ± 5.92% | 4.65× |
| DB2 | Intact-limb (40 subjects, 17 classes) | 13.08 ± 3.05% | 5.88% | 2.22× | 53.20 ± 9.32% | 4.07× |
| DB3 | Amputee (11 subjects, 17 classes) | 6.63 ± 1.22% | 5.88% | 1.13× (n.s.) | 39.88 ± 11.39% | 6.02× |

- **Within-subject accuracy exceeds cross-subject (LOSO) accuracy by 4–6×** on identical data, identical pipeline (Wilcoxon p ≤ 0.001, Cohen's d ≥ 2.96 on all three databases).
- **CORAL significantly worsened accuracy** on DB7 (Cohen's d = 0.92) and DB2 (d = 1.11), rather than improving it (Wilcoxon p < 0.001 on both). TCA and SA were statistically indistinguishable from the unadapted baseline. On DB3, all five treatments cluster near chance with no significant differences among them (Friedman p = 0.443).
- **Mechanistic root cause:** a covariance-eigenvalue analysis of the PPV feature space reveals extreme redundancy among random convolutional kernels (condition number ≈ 9.5×10¹¹; 97.2% of eigenvalues below 10× the regularization floor). This causes CORAL's whitening–recoloring transform to amplify noise catastrophically along near-singular directions, stretching bounded [0,1] PPV values into a range exceeding ±6.
- **Domain-shift metrics confirm this directly:** CORAL *increased* between-subject covariance distance by 9.1× (DB7), 6.8× (DB2), and 1.3× (DB3) — the opposite of what a domain-alignment method is supposed to do.
- **DB3 (amputee) resists all five treatments uniformly** rather than showing one method underperform another, which is more consistent with high between-subject anatomical/electrode-placement heterogeneity in amputee sEMG than with a representation-level failure specific to any one method (see paper Section 3.9 for the full argument and its limits).
- A window-size ablation shows accuracy increases monotonically through 800 ms with no plateau reached (Section 3.7), while a training-sample-size ablation (10k/25k/50k) shows the main result is **not** an artifact of insufficient training data (Section 3.8).

Full statistical tests (Friedman, Wilcoxon + effect sizes, Nemenyi), the eigenvalue/distortion diagnostics, domain-shift metrics, kernel-importance analysis, and both ablations are reported in the paper and reproduced under [`outputs/`](#repository-structure) below.

## Repository Structure

```
MiniROCKET-sEMG-CrossSubject-DomainAdaptation/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── .gitattributes
├── setup_and_run.bat                # Windows convenience launcher (venv + menu)
│
├── config.py                        # Paths, dataset metadata, MiniROCKET/DA/classifier hyperparameters
├── data_loader.py                   # NinaPro DB2 / DB3 / DB7 streaming .mat loader
├── minirocket.py                    # MiniROCKET transform (10,000 kernels, dilations {1,2,4,8,16}, PPV pooling)
├── domain_adaptation.py             # Feature centering, CORAL, TCA, Subspace Alignment
├── statistical_utils.py             # Friedman, Wilcoxon+Holm, Nemenyi, Cohen's d helpers
│
├── p01a_run_main_loso_benchmark.py            # Main LOSO benchmark: 5 methods × DB2/DB3/DB7 → Table 2
├── p01b_run_within_subject_baseline.py        # Matched within-subject reference ceiling → Table 3
├── p02_run_window_and_sample_ablation.py      # Window-size + training-sample-size ablations → Tables 10-11
├── p03a_bridge_result_filenames.py            # Bridges p01a's output filenames/JSON keys to what stages 04+ expect
├── p03b_clean_stale_kernel_importance_files.py # Removes corrupted/dimension-mismatched kernel-importance .npz files
├── p04a_extract_kernel_importance_data.py     # Extracts raw per-kernel importance data (fast variant) → .npz
├── p04b_compute_domain_shift_metrics.py       # Covariance Frobenius distance, source vs. target → Table 6
├── p04c_run_coral_distortion_diagnostic.py    # Eigenvalue spectrum + before/after distortion diagnostic → Tables 7-8
├── p05_analyze_kernel_importance.py           # Kernel-importance tables/figures from p04a's .npz data → Table 9, Fig. 6-7
├── p06_run_statistical_analysis.py            # Friedman / Wilcoxon / Nemenyi / Cohen's d → Table 4, 5, S1-S8
├── p06b_verify_and_clean_perclass_f1_table.py # Verifies/cleans a phantom-class artifact in the per-class F1 table
├── p07_generate_all_figures.py                # All figures: accuracy, significance heatmaps, eigenvalue spectra, confusion matrices, t-SNE
│
├── run_full_pipeline.py             # Orchestrator: runs p01a → p07 end-to-end in one command
├── run_pipeline_from_stage03.py     # Orchestrator: resumes from the bridge/clean step through p07
│                                     #   (the up-to-date step list for stages 03+; see script docstring)
│
├── tools/                           # Standalone diagnostics, not part of the numbered pipeline
│   ├── check_db7_labels_against_raw_data.py
│   └── db7_label_diagnostic.txt
│
├── dev_history/                     # Archived, already-applied one-time patches (see dev_history/README.md)
│
├── logs/                            # Timestamped run logs
└── outputs/
    ├── results/                    # Raw per-fold JSON results, one per (database, method)
    ├── tables/                     # All CSV/TeX tables reported in the paper (Table 2 through TableS8)
    ├── figures/                    # All figures reported in the paper
    ├── features/                   # Extracted kernel-importance arrays (.npz), consumed by p05
    ├── ablation/                   # Window-size and training-sample-size ablation results
    └── checkpoints/                # Per-fold resume checkpoints for long LOSO runs
```

> Pipeline scripts are prefixed `p` + a phase number (`01` = main results, `02` = ablations, `03` = housekeeping bridge/cleanup, `04` = mechanistic diagnostics, `05` = kernel-importance analysis, `06` = statistics, `07` = figures) rather than by development date. The `p` prefix is not decorative: `p02` and `p07` both import functions directly from `p01a` / `p04b` at runtime, and Python module names cannot start with a digit — a bare `01a_...` filename would have made those imports a `SyntaxError`.

## Pipeline (7 phases, 365 LOSO folds)

1. **Main LOSO benchmark + within-subject baseline** (`p01a`, `p01b`) — the 15 experimental conditions (5 methods × 3 databases) and the matched within-subject reference ceiling.
2. **Ablations** (`p02`) — window size (100–800 ms) and training-sample-size (10k/25k/50k) robustness checks.
3. **Housekeeping** (`p03a`, `p03b`) — bridges filename/JSON-key differences between stage 1's output and what stages 4+ expect, and removes any corrupted kernel-importance files (a known issue with TCA/SA, which project to a different dimensionality).
4. **Mechanistic diagnostics** (`p04a`, `p04b`, `p04c`) — kernel-importance data extraction, domain-shift (covariance Frobenius distance) metrics, and the CORAL eigenvalue/distortion diagnostic that explains *why* CORAL fails on PPV features specifically.
5. **Kernel-importance analysis** (`p05`) — builds the final per-dilation importance tables and figures from stage 4's extracted data.
6. **Statistics** (`p06`, `p06b`) — Friedman omnibus test, pairwise Wilcoxon + Holm–Šidák, Nemenyi post-hoc, Cohen's d, and a data-integrity check on the per-class F1 table.
7. **Figures** (`p07`) — every figure reported in the paper.

## Installation & Quick Start

```bash
git clone https://github.com/Qussai-BME/MiniROCKET-sEMG-CrossSubject-DomainAdaptation.git
cd MiniROCKET-sEMG-CrossSubject-DomainAdaptation
pip install -r requirements.txt
```

Edit `config.py`'s `DB_PATHS` to point at your local NinaPro DB2/DB3/DB7 download (see Data Availability below).

### Run the full pipeline
```bash
python run_full_pipeline.py --all
```
Or run each phase individually — see [Repository Structure](#repository-structure) above, or pass `--help` to any script for its full set of flags. On Windows, `setup_and_run.bat` provides an interactive menu that wraps the same commands.

### Use pre-computed results directly (no re-run needed)
Everything reported in the paper is already available under [`outputs/`](#repository-structure) — per-fold JSON results, all tables (CSV + TeX), all figures, and the extracted kernel-importance data.

## Data Availability

NinaPro databases DB2, DB3, and DB7 are publicly available at [https://ninapro.hevs.ch](https://ninapro.hevs.ch). This repository does not redistribute the raw data. For DB2/DB3, only Exercise B (17 movements) is used, to avoid a movement-label collision that arises when different exercise files are concatenated without a per-exercise label offset — a pitfall identified during pipeline validation (see `tools/` and `dev_history/`).

## Environment

| Component | Version |
|---|---|
| Python | 3.10+ |
| NumPy | ≥1.21 |
| SciPy | ≥1.7 |
| scikit-learn | ≥1.0 |
| statsmodels | ≥0.13 |
| h5py | ≥3.0 |

See [`requirements.txt`](requirements.txt) for the full, versioned dependency list.

## Limitations (condensed — see paper §4.3 for the complete list)

- The within-subject baseline uses a single held-out 80/20 split per subject, not internal k-fold CV, so the reported 4–6× gap is approximate rather than a precisely bounded estimate.
- "Feature centering" is numerically identical to "raw" throughout this study, because the ridge classifier's downstream z-score standardization already performs mean-centering. Results attributed to centering in Tables 2, 4, and 5 should be read as a null replicate of raw, not an independent comparison.
- The kernel-importance analysis (Section 3.6) and the eigenvalue/distortion diagnostic (Section 3.5) both used reduced-scale configurations for computational tractability and have not been replicated at the full 10,000-kernel, 22-subject scale.
- Domain-shift Frobenius-distance magnitudes for TCA are not directly comparable to the other methods due to differing feature-space dimensionality.
- The amputee-specific failure mode on DB3 (Section 3.9) is offered as our leading interpretation, not a confirmed mechanism — we did not directly measure anatomical or electrode-placement variability in this study.

## Citation

```bibtex
@article{adlbi2026minirocket,
  title   = {Why MiniROCKET PPV Features Resist Domain Adaptation: A Mechanistic Study of Cross-Subject sEMG Gesture Recognition Failure},
  author  = {Adlbi, Qussai and Darwich, Mohamad Ayham},
  journal = {Under review},
  year    = {2026}
}
```
This entry will be updated with the final journal, volume, and DOI once the paper is published.

## License

MIT — see [LICENSE](LICENSE).

## What this is not

- Not a claim that domain adaptation is useless for sEMG in general — only that four standard *linear covariance-alignment* methods fail specifically on MiniROCKET's high-redundancy PPV representation.
- Not a full anatomical or electrode-registration study of amputee sEMG variability (Section 3.9's DB3 interpretation is a hypothesis, not a directly measured finding).
- Not a benchmark against hand-crafted time/frequency-domain features under the same protocol — see the companion repository [sEMG-Zero-Calibration-LOSO-Benchmark](https://github.com/Qussai-BME/sEMG-Zero-Calibration-LOSO-Benchmark) for that comparison, which motivated the present study's choice to isolate MiniROCKET specifically.
