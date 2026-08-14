"""
[ARCHIVED — ALREADY APPLIED, kept for provenance only]
This one-time patch has already been run against p07_generate_all_figures.py
(see its "Movement-name helpers (added by ...)" marker comment). Re-running
it is safe (idempotent) but not necessary. See dev_history/README.md.
------------------------------------------------------------------------------
dev_history/apply_confusion_matrix_labels_patch_already_applied.py
===========================
Automatically patches p07_generate_all_figures.py to:
  1. Add movement-name tick labels to the confusion matrix axes
  2. Add a hard safety check: if the configured name list doesn't match the
     actual number of classes exactly, it REFUSES to apply names and falls
     back to plain numeric indices (today's behavior) rather than risk
     mislabeling a class.
  3. Correct the "(Showing first N of M classes)" caption to use the true,
     verified movement count (40 for DB7) instead of whatever stale
     n_classes value happens to be stored in an old results JSON.

Creates p07_generate_all_figures.py.bak before touching anything. Safe to run
once; running it again will detect the patch is already applied and do
nothing.

USAGE:
    python dev_history/apply_confusion_matrix_labels_patch_already_applied.py
"""
import re
import shutil
import sys
import os

TARGET_PATH = "p07_generate_all_figures.py"

HELPER_FUNCTIONS = '''

# ============================================================
# Movement-name helpers (added by dev_history/apply_confusion_matrix_labels_patch_already_applied.py)
# ============================================================
def get_movement_names(db_key):
    """
    Returns the (rest + movement) name list for a database key, or None if
    unavailable. DB2 and DB3 share the same Exercise B protocol and
    therefore the same name list.
    """
    if db_key == "db7":
        return getattr(config, "DB7_MOVEMENT_NAMES", None)
    if db_key in ("db2", "db3"):
        return getattr(config, "DB2_MOVEMENT_NAMES", None)
    return None


def true_movement_count(db_key, fallback_n_classes):
    """
    Returns the verified true movement count (e.g. 40 for DB7) derived from
    the configured name list, or falls back to whatever n_classes was
    passed in (e.g. from an old results JSON) if no name list is available.
    """
    names = get_movement_names(db_key)
    if names is None:
        return fallback_n_classes
    return len(names) - 1  # drop "rest"


def safe_class_labels(db_key, n_classes, max_show):
    """
    Returns a list of `max_show` tick labels for the confusion matrix axes.
    Falls back to plain numeric indices and prints a warning if the
    configured name list doesn't match n_classes in length — this is the
    guard that prevents ever silently mislabeling a class.
    """
    names = get_movement_names(db_key)
    numeric_fallback = [str(i + 1) for i in range(max_show)]

    if names is None:
        print(f"  [INFO] No movement-name list configured for '{db_key}' — "
              f"using numeric class indices.", flush=True)
        return numeric_fallback

    non_rest_names = names[1:]  # drop "rest"
    if len(non_rest_names) != n_classes:
        print(f"  [WARNING] Movement-name list for '{db_key}' has "
              f"{len(non_rest_names)} entries but this run has n_classes="
              f"{n_classes}. REFUSING to apply names (would risk "
              f"mislabeling classes) — falling back to numeric indices. "
              f"If n_classes looks stale (e.g. 41 for DB7), that's expected "
              f"until you rerun the LOSO pipeline with the corrected "
              f"config.py; the plot itself is still correct for the "
              f"classes actually shown.", flush=True)
        return numeric_fallback

    labels = []
    for i in range(max_show):
        full_name = non_rest_names[i]
        short = full_name if len(full_name) <= 22 else full_name[:19] + "..."
        labels.append(f"{i + 1}: {short}")
    return labels

'''


def fail(msg):
    print(f"[FAIL] {msg}")
    sys.exit(1)


def main():
    if not os.path.exists(TARGET_PATH):
        fail(f"{TARGET_PATH} not found in the current directory. "
             f"Run this script from the project folder.")

    with open(TARGET_PATH, encoding="utf-8") as f:
        content = f.read()

    if "Movement-name helpers (added by dev_history/apply_confusion_matrix_labels_patch_already_applied.py)" in content:
        print("[INFO] Patch already applied. Nothing to do.")
        return

    shutil.copy(TARGET_PATH, TARGET_PATH + ".bak")
    print(f"[OK] Backed up {TARGET_PATH} -> {TARGET_PATH}.bak")

    # --- Insert helper functions right after "import config" ---
    anchor = "import config\n"
    if content.count(anchor) == 0:
        fail('Could not find the line "import config" to anchor the helper '
             'functions after. Insert HELPER_FUNCTIONS manually after your '
             'config import.')
    content = content.replace(anchor, anchor + HELPER_FUNCTIONS, 1)

    # --- Patch the imshow/label block inside plot_confusion_matrices ---
    old_block = """        # رسم مصفوفة الارتباك
        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(cm_show, cmap='Blues', vmin=0, vmax=1, interpolation='nearest')
        cbar = plt.colorbar(im, ax=ax, label='Normalized Frequency', shrink=0.8)

        ax.set_xlabel('Predicted Class', fontsize=12)
        ax.set_ylabel('True Class', fontsize=12)

        db_display = config.DB_META.get(db_key, {}).get('display_name', db_key)
        method_label = config.METHOD_LABELS.get(best_method, best_method)
        ax.set_title(f'Aggregated Confusion Matrix — {db_display}\\n'
                     f'Best Method: {method_label} ({best_acc*100:.1f}%)',
                     fontweight='bold', fontsize=12)

        if max_show < n_classes:
            ax.text(
                0.5, -0.08,
                f'(Showing first {max_show} of {n_classes} classes)',
                transform=ax.transAxes, ha='center', va='top',
                fontsize=9, style='italic'
            )"""

    new_block = """        # رسم مصفوفة الارتباك
        fig, ax = plt.subplots(figsize=(12, 10))

        im = ax.imshow(cm_show, cmap='Blues', vmin=0, vmax=1, interpolation='nearest')
        cbar = plt.colorbar(im, ax=ax, label='Normalized Frequency', shrink=0.8)

        # Movement-name tick labels, with a hard safety fallback to numeric
        # indices if the configured name list doesn't match n_classes exactly.
        true_n = true_movement_count(db_key, n_classes)
        tick_labels = safe_class_labels(db_key, true_n, max_show)
        ax.set_xticks(range(max_show))
        ax.set_yticks(range(max_show))
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=7, ha='center')
        ax.set_yticklabels(tick_labels, fontsize=7)

        ax.set_xlabel('Predicted Class', fontsize=12, labelpad=10)
        ax.set_ylabel('True Class', fontsize=12)

        db_display = config.DB_META.get(db_key, {}).get('display_name', db_key)
        method_label = config.METHOD_LABELS.get(best_method, best_method)
        ax.set_title(f'Aggregated Confusion Matrix — {db_display}\\n'
                     f'Best Method: {method_label} ({best_acc*100:.1f}%)',
                     fontweight='bold', fontsize=12)

        if max_show < true_n:
            ax.text(
                0.5, -0.08,
                f'(Showing first {max_show} of {true_n} classes)',
                transform=ax.transAxes, ha='center', va='top',
                fontsize=9, style='italic'
            )"""

    if content.count(old_block) == 0:
        fail('Could not find the exact confusion-matrix plotting block to '
             'patch. The file may have been edited since this script was '
             'written. Paste me the current content of plot_confusion_matrices '
             'and I will regenerate this script to match exactly.')
    if content.count(old_block) > 1:
        fail('Found more than one match for the plotting block — refusing '
             'to guess which one. Send me the file and I will patch it by hand.')

    content = content.replace(old_block, new_block, 1)

    with open(TARGET_PATH, "w", encoding="utf-8") as f:
        f.write(content)

    print("[OK] p07_generate_all_figures.py patched successfully.")
    print()
    print("Now run:  python p07_generate_all_figures.py --skip_tsne")
    print("Watch for any line starting with [WARNING] in the output — if you")
    print("see one, copy the full line and send it to me before trusting the")
    print("resulting figure.")


if __name__ == "__main__":
    main()
