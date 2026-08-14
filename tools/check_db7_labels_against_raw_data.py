"""
tools/check_db7_labels_against_raw_data.py — Standalone diagnostic: how many movement classes does
DB7 ACTUALLY contain in your raw data, independent of any documentation claim
or existing pipeline assumption?

WHY THIS EXISTS:
  config.py currently declares n_movements=41 for DB7. Two independent official
  sources (ninapro.hevs.ch's own DB7 page, and the Krasoulis et al. 2017 paper's
  own Methods text) state the protocol has 40 movements (+ rest). This script
  settles the question empirically against YOUR actual downloaded files, rather
  than trusting either the code or the documentation blindly.

WHAT IT DOES:
  Loads every DB7 .mat file directly with scipy.io.loadmat (bypassing the
  windowing/feature pipeline entirely), reads the raw 'restimulus' (or
  'stimulus') label column, and reports:
    - unique label values found, per subject and in aggregate
    - min/max label value
    - total distinct non-rest movement count
    - a direct comparison against config.py's declared n_movements
    - a per-subject breakdown, so you can also see if any subject is
      missing specific movements (as documented for subject 21)

USAGE:
    python tools/check_db7_labels_against_raw_data.py

OUTPUT:
  Prints a full report to stdout AND writes it to db7_label_diagnostic.txt
  so you can attach it to notes / share it if anything looks off.
"""
import os
import sys
import glob
import numpy as np
import scipy.io

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # this script now lives in tools/
sys.path.insert(0, PROJECT_ROOT)
import config


def load_raw_labels(filepath):
    """Load only the label column from a .mat file, as cheaply as possible."""
    try:
        data = scipy.io.loadmat(filepath)
    except Exception as e:
        print(f"  [ERROR] Could not load {filepath}: {e}")
        return None
    for key in ('restimulus', 'stimulus'):
        if key in data:
            return np.asarray(data[key]).ravel(), key
    print(f"  [WARNING] No 'restimulus'/'stimulus' key in {filepath}. "
          f"Available keys: {[k for k in data.keys() if not k.startswith('__')]}")
    return None, None


def find_db7_files():
    db_path = config.DB_PATHS.get("db7")
    if db_path is None or not os.path.isdir(db_path):
        print(f"[ERROR] DB7 path not found or not configured: {db_path}")
        print("        Edit config.DB_PATHS['db7'] to point at your local DB7 folder, then rerun.")
        sys.exit(1)

    mat_files = sorted(glob.glob(os.path.join(db_path, "**", "*.mat"), recursive=True))
    if not mat_files:
        mat_files = sorted(glob.glob(os.path.join(db_path, "*.mat")))
    if not mat_files:
        print(f"[ERROR] No .mat files found under {db_path}")
        sys.exit(1)
    return mat_files


def subject_id_from_filename(fname):
    """Best-effort subject-id extraction (S1_..., S01_..., subject1_..., etc.)."""
    base = os.path.basename(fname)
    digits = ''.join(ch if ch.isdigit() else ' ' for ch in base).split()
    return digits[0] if digits else base


def main():
    lines = []

    def log(msg=""):
        print(msg, flush=True)
        lines.append(msg)

    log("=" * 72)
    log("DB7 RAW LABEL DIAGNOSTIC")
    log("=" * 72)
    log(f"Config currently declares: n_movements={config.DB_META['db7']['n_movements']}, "
        f"movement_ids={config.DB_META['db7']['movement_ids'][0]}..{config.DB_META['db7']['movement_ids'][-1]}")
    log("")

    mat_files = find_db7_files()
    log(f"Found {len(mat_files)} .mat file(s) under {config.DB_PATHS['db7']}")
    log("")

    per_subject_unique = {}
    all_labels_seen = set()
    label_key_used = None

    for fpath in mat_files:
        sid = subject_id_from_filename(fpath)
        result = load_raw_labels(fpath)
        if result is None or result[0] is None:
            continue
        labels, key = result
        label_key_used = label_key_used or key
        uniq = set(int(x) for x in np.unique(labels))
        per_subject_unique.setdefault(sid, set()).update(uniq)
        all_labels_seen.update(uniq)
        log(f"  Subject {sid:>4} ({os.path.basename(fpath)}): "
            f"{len(uniq)} unique labels, range [{min(uniq)}, {max(uniq)}]  (key='{key}')")

    log("")
    log("=" * 72)
    log("AGGREGATE RESULT")
    log("=" * 72)

    if not all_labels_seen:
        log("[ERROR] No labels were successfully read from any file. Check the .mat "
            "file structure and the label key names above.")
        return

    sorted_labels = sorted(all_labels_seen)
    has_rest = 0 in sorted_labels
    non_rest = [l for l in sorted_labels if l != 0]

    log(f"Label key used: '{label_key_used}'")
    log(f"All unique label values found across every subject: {sorted_labels}")
    log(f"Rest label (0) present: {has_rest}")
    log(f"Number of DISTINCT NON-REST movement labels: {len(non_rest)}")
    log(f"Min non-rest label: {min(non_rest) if non_rest else 'n/a'}")
    log(f"Max non-rest label: {max(non_rest) if non_rest else 'n/a'}")
    log("")

    declared = config.DB_META['db7']['n_movements']
    actual = len(non_rest)

    log("-" * 72)
    if actual == declared:
        log(f"RESULT: MATCH. config.py declares n_movements={declared}; your raw data "
            f"also contains exactly {actual} distinct non-rest movement labels.")
        log("ACTION: No change needed to config.py's n_movements for DB7.")
    else:
        log(f"RESULT: MISMATCH. config.py declares n_movements={declared}, but your raw "
            f"data contains {actual} distinct non-rest movement labels.")
        log("ACTION: This needs manual review before changing anything in the paper —")
        log("        do not silently 'correct' n_movements without understanding why")
        log("        the two disagree (check for gaps in the label range, an extra")
        log("        transition/rest-like code, or a per-subject anomaly below).")
    log("-" * 72)
    log("")

    log("Per-subject unique non-rest movement counts (look for outliers, e.g. subject 21):")
    for sid in sorted(per_subject_unique, key=lambda x: (len(x), x)):
        uniq = per_subject_unique[sid]
        n_non_rest = len([l for l in uniq if l != 0])
        log(f"  Subject {sid:>4}: {n_non_rest} non-rest movements")

    out_path = os.path.join(SCRIPT_DIR, "db7_label_diagnostic.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log("")
    log(f"Full report written to: {out_path}")


if __name__ == "__main__":
    main()