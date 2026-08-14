# Development History (archived, already applied)

This folder preserves the one-time patch scripts that fixed two issues
discovered after the first pipeline run, plus the reference notes that
preceded them. All of them have **already been applied** — they are kept
here for transparency/provenance, not because anything still needs running.

| File | What it did | Status |
|---|---|---|
| `apply_config_fix_already_applied.py` | Corrected DB7's declared movement count (41 → 40) and replaced the DB2/DB7 movement-name lists in `config.py` with the verified names. Created `config.py.bak` as a safety copy before editing. | **Already applied.** `config.py` already reflects the fix. |
| `apply_confusion_matrix_labels_patch_already_applied.py` | Added movement-name tick labels (with a hard safety fallback to numeric indices on any length mismatch) to the confusion-matrix plots in `p07_generate_all_figures.py`. Created a `.bak` safety copy before editing. | **Already applied.** `p07_generate_all_figures.py` already contains the patched plotting code. |
| `confusion_matrix_labels_patch_reference_notes.py` | The manual, human-readable version of the edit that `apply_confusion_matrix_labels_patch_already_applied.py` later automated. | Superseded by the script above — kept as documentation of the reasoning. |
| `movement_names_reference_notes.py` | The manual, human-readable version of the movement-name lists that `apply_config_fix_already_applied.py` later applied automatically. | Superseded by the script above — kept as documentation. |

**If you are trying to reproduce the paper's results, you do not need to run
anything in this folder** — the fixes are already baked into `config.py` and
`p07_generate_all_figures.py`. Start from the main [README](../README.md)
instead.
