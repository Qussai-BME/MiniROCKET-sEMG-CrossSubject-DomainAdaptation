"""
[ARCHIVED — SUPERSEDED, kept for provenance only]
The edit documented below was later automated by
dev_history/apply_confusion_matrix_labels_patch_already_applied.py, which has
already been run. This file is reference documentation only. See
dev_history/README.md.
------------------------------------------------------------------------------
dev_history/confusion_matrix_labels_patch_reference_notes.py
=====================================
This is not a script to run — it documents the exact edit to make to
p07_generate_all_figures.py, and contains the full replacement code to paste in.

WHY A SEPARATE FILE INSTEAD OF EDITING DIRECTLY:
  You should run tools/check_db7_labels_against_raw_data.py FIRST and read its verdict before
  touching p07_generate_all_figures.py. The code below is written to be SAFE
  regardless of that verdict — it will never mislabel a class. If the name
  list and the actual data disagree in length, it prints a loud warning and
  falls back to plain numeric indices (today's behavior) instead of quietly
  pairing the wrong name with the wrong class.

WHAT CHANGED VS. THE ORIGINAL plot_confusion_matrices:
  1. A new helper, get_movement_names(db_key), maps a database key to its
     name list (DB7_MOVEMENT_NAMES for db7; DB2_MOVEMENT_NAMES for db2/db3,
     since both use the identical Exercise B protocol).
  2. Before applying any name to an axis, the length of the name list
     (excluding "rest") is checked against n_classes for that specific run.
     A mismatch prints a clear, impossible-to-miss warning and the plot
     falls back to numeric tick labels — exactly like before this patch.
  3. When names ARE applied, long names are truncated for the tick label
     (full names remain available via the Supplementary Material lookup
     table) and rotated 90° for legibility with up to 20 classes shown.

===========================================================================
STEP 1 — add this near the top of p07_generate_all_figures.py, after the
`import config` line (or anywhere before plot_confusion_matrices is defined):
===========================================================================

def get_movement_names(db_key):
    \"\"\"
    Returns the (rest + movement) name list for a database key, or None if
    unavailable. DB2 and DB3 share the same Exercise B protocol and therefore
    the same name list.
    \"\"\"
    if db_key == "db7":
        return getattr(config, "DB7_MOVEMENT_NAMES", None)
    if db_key in ("db2", "db3"):
        return getattr(config, "DB2_MOVEMENT_NAMES", None)
    return None


def safe_class_labels(db_key, n_classes, max_show):
    \"\"\"
    Returns a list of `max_show` tick labels for the confusion matrix axes.
    Falls back to plain numeric indices (current behavior) and prints a
    warning if the configured name list doesn't match n_classes in length —
    this is the guard that prevents ever silently mislabeling a class.
    \"\"\"
    names = get_movement_names(db_key)
    numeric_fallback = [str(i) for i in range(max_show)]

    if names is None:
        print(f"  [INFO] No movement-name list configured for '{db_key}' — "
              f"using numeric class indices.", flush=True)
        return numeric_fallback

    non_rest_names = names[1:]  # drop "rest"
    if len(non_rest_names) != n_classes:
        print(f"  [WARNING] Movement-name list for '{db_key}' has "
              f"{len(non_rest_names)} entries but this run has n_classes="
              f"{n_classes}. REFUSING to apply names (would risk mislabeling "
              f"classes) — falling back to numeric indices. Run "
              f"tools/check_db7_labels_against_raw_data.py and reconcile config.py before retrying.",
              flush=True)
        return numeric_fallback

    labels = []
    for i in range(max_show):
        full_name = non_rest_names[i]
        short = full_name if len(full_name) <= 22 else full_name[:19] + "..."
        labels.append(f"{i+1}: {short}")
    return labels


===========================================================================
STEP 2 — inside plot_confusion_matrices, replace this block:
===========================================================================

    OLD:
    ----
        # تحديد عدد الأصناف المعروضة (حد أقصى 20 للتشغيل السريع)
        max_show = min(20, n_classes)
        cm_show = agg_cm[:max_show, :max_show]

        # رسم مصفوفة الارتباك
        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(cm_show, cmap='Blues', vmin=0, vmax=1, interpolation='nearest')
        cbar = plt.colorbar(im, ax=ax, label='Normalized Frequency', shrink=0.8)

        ax.set_xlabel('Predicted Class', fontsize=12)
        ax.set_ylabel('True Class', fontsize=12)

    NEW:
    ----
        # تحديد عدد الأصناف المعروضة (حد أقصى 20 للتشغيل السريع)
        max_show = min(20, n_classes)
        cm_show = agg_cm[:max_show, :max_show]

        # رسم مصفوفة الارتباك
        fig, ax = plt.subplots(figsize=(12, 10))

        im = ax.imshow(cm_show, cmap='Blues', vmin=0, vmax=1, interpolation='nearest')
        cbar = plt.colorbar(im, ax=ax, label='Normalized Frequency', shrink=0.8)

        # Movement-name tick labels, with a hard safety fallback to numeric
        # indices if the configured name list doesn't match n_classes exactly.
        tick_labels = safe_class_labels(db_key, n_classes, max_show)
        ax.set_xticks(range(max_show))
        ax.set_yticks(range(max_show))
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=7, ha='center')
        ax.set_yticklabels(tick_labels, fontsize=7)

        ax.set_xlabel('Predicted Class', fontsize=12, labelpad=10)
        ax.set_ylabel('True Class', fontsize=12)

===========================================================================
NOTHING ELSE in the function needs to change — the title, the "(Showing
first N of M classes)" annotation, the spine removal, and the save call
are all unaffected by this patch.
===========================================================================

REMINDER — order of operations:
  1. Run tools/check_db7_labels_against_raw_data.py. Read its verdict.
  2. If it reports a mismatch for DB7, resolve config.py's n_movements/
     movement_ids/DB7_MOVEMENT_NAMES BEFORE running day6 again — do not
     patch day6 first and hope the safety fallback "handles it silently."
     The fallback exists to protect you from a bad plot, not to replace
     fixing the actual discrepancy.
  3. Only after config.py is confirmed consistent with the real data,
     apply the STEP 1 / STEP 2 edits above and regenerate the figures.
"""