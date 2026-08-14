"""
[ARCHIVED — ALREADY APPLIED, kept for provenance only]
This one-time patch has already been run against config.py (see
config.py's "Safety checks added by ..." marker comment, and the fact
that DB7 is already declared with 40 movements). Re-running it is safe
(idempotent) but not necessary. See dev_history/README.md.
------------------------------------------------------------------------------
dev_history/apply_config_fix_already_applied.py
==========================
Automatically patches config.py:
  1. DB7's n_movements: 41 -> 40
  2. DB7's movement_ids: range(1,42) -> range(1,41)
  3. DB2_MOVEMENT_NAMES: replaced with the verified 17-movement Exercise B list
  4. DB7_MOVEMENT_NAMES: replaced with the verified 40-movement (Exercise B+C) list

Creates config.py.bak before touching anything. Safe to run once; running it
again will detect the fix is already applied and do nothing.

USAGE:
    python dev_history/apply_config_fix_already_applied.py
"""
import re
import shutil
import sys
import os

CONFIG_PATH = "config.py"


def fail(msg):
    print(f"[FAIL] {msg}")
    sys.exit(1)


def main():
    if not os.path.exists(CONFIG_PATH):
        fail(f"{CONFIG_PATH} not found in the current directory. "
             f"Run this script from the project folder (same folder as config.py).")

    with open(CONFIG_PATH, encoding="utf-8") as f:
        content = f.read()

    already_done = ('"n_movements":   40,' in content and
                     '"movement_ids":  list(range(1, 41))' in content)
    if already_done:
        print("[INFO] config.py already has n_movements=40 for DB7. Nothing to do here.")
        print("       (If DB7_MOVEMENT_NAMES/DB2_MOVEMENT_NAMES still look wrong, "
              "delete them manually and rerun this script.)")
        return

    shutil.copy(CONFIG_PATH, CONFIG_PATH + ".bak")
    print(f"[OK] Backed up {CONFIG_PATH} -> {CONFIG_PATH}.bak")

    changes_made = []

    # --- Fix 1: n_movements 41 -> 40 (inside the "db7" block specifically) ---
    db7_block_pattern = re.compile(r'("db7":\s*\{.*?\n\s*\},)', re.DOTALL)
    m = db7_block_pattern.search(content)
    if not m:
        fail('Could not locate the "db7": { ... } block in config.py. '
             'The file may have been restructured — apply the fix manually.')
    db7_block = m.group(1)
    new_db7_block = db7_block

    new_db7_block, n1 = re.subn(r'"n_movements":\s*41,', '"n_movements":   40,', new_db7_block)
    new_db7_block, n2 = re.subn(
        r'"movement_ids":\s*list\(range\(1,\s*42\)\)[^\n]*',
        '"movement_ids":  list(range(1, 41)),   # 1..40 (was incorrectly 1..41)',
        new_db7_block)
    if n1 == 0 or n2 == 0:
        fail('Found the "db7" block but could not find the expected n_movements=41 '
             'or movement_ids=range(1,42) lines inside it. Check for manual edits '
             'and apply the fix by hand, comparing against the chat instructions.')

    content = content[:m.start()] + new_db7_block + content[m.end():]
    changes_made.append("db7.n_movements: 41 -> 40")
    changes_made.append("db7.movement_ids: range(1,42) -> range(1,41)")

    # --- Fix 2: DB2_MOVEMENT_NAMES (used by DB2 and DB3) ---
    db2_names = '''DB2_MOVEMENT_NAMES = [
    "rest",
    "Thumb up",
    "Extension of index and middle, flexion of the others",
    "Flexion of ring and little finger, extension of the others",
    "Thumb opposing base of little finger",
    "Abduction of all fingers",
    "Fingers flexed together in fist",
    "Pointing index",
    "Adduction of extended fingers",
    "Wrist supination (axis: middle finger)",
    "Wrist pronation (axis: middle finger)",
    "Wrist supination (axis: little finger)",
    "Wrist pronation (axis: little finger)",
    "Wrist flexion",
    "Wrist extension",
    "Wrist radial deviation",
    "Wrist ulnar deviation",
    "Wrist extension with closed hand",
]'''

    db2_pattern = re.compile(r'DB2_MOVEMENT_NAMES\s*=\s*\[.*?\n\]', re.DOTALL)
    if not db2_pattern.search(content):
        fail('Could not locate DB2_MOVEMENT_NAMES = [ ... ] in config.py.')
    content = db2_pattern.sub(db2_names.replace('\\', '\\\\'), content, count=1)
    changes_made.append("DB2_MOVEMENT_NAMES: replaced with verified 17-movement Exercise B list")

    # --- Fix 3: DB7_MOVEMENT_NAMES (Exercise B + Exercise C = 40) ---
    db7_names = '''DB7_MOVEMENT_NAMES = [
    "rest",
    "Thumb up",
    "Extension of index and middle, flexion of the others",
    "Flexion of ring and little finger, extension of the others",
    "Thumb opposing base of little finger",
    "Abduction of all fingers",
    "Fingers flexed together in fist",
    "Pointing index",
    "Adduction of extended fingers",
    "Wrist supination (axis: middle finger)",
    "Wrist pronation (axis: middle finger)",
    "Wrist supination (axis: little finger)",
    "Wrist pronation (axis: little finger)",
    "Wrist flexion",
    "Wrist extension",
    "Wrist radial deviation",
    "Wrist ulnar deviation",
    "Wrist extension with closed hand",
    "Large diameter grasp",
    "Small diameter grasp",
    "Fixed hook grasp",
    "Index finger extension grasp",
    "Medium wrap",
    "Ring grasp",
    "Prismatic four fingers grasp",
    "Stick grasp",
    "Writing tripod grasp",
    "Power grasp",
    "Three-finger sphere grasp",
    "Precision sphere grasp",
    "Tripod grasp",
    "Prismatic pinch",
    "Tip pinch",
    "Quadpod grasp",
    "Lateral grasp",
    "Parallel extension grasp",
    "Parallel flexion grasp",
    "Power disk grasp",
    "Open a bottle (tripod grasp)",
    "Turn a screw (stick grasp)",
    "Cut something (index finger extension grasp)",
]'''

    db7_pattern = re.compile(r'DB7_MOVEMENT_NAMES\s*=\s*\[.*?\n\]', re.DOTALL)
    if not db7_pattern.search(content):
        fail('Could not locate DB7_MOVEMENT_NAMES = [ ... ] in config.py.')
    content = db7_pattern.sub(db7_names, content, count=1)
    changes_made.append("DB7_MOVEMENT_NAMES: replaced with verified 40-movement (Exercise B+C) list")

    # --- Append safety asserts at end of file (idempotent) ---
    assert_block = '''

# --- Safety checks added by dev_history/apply_config_fix_already_applied.py ---
assert len(DB2_MOVEMENT_NAMES) == 18, \\
    f"DB2_MOVEMENT_NAMES has {len(DB2_MOVEMENT_NAMES)} entries, expected 18 (rest + 17)"
assert len(DB7_MOVEMENT_NAMES) == 41, \\
    f"DB7_MOVEMENT_NAMES has {len(DB7_MOVEMENT_NAMES)} entries, expected 41 (rest + 40)"
assert DB_META["db7"]["n_movements"] == 40, \\
    "DB_META['db7']['n_movements'] should be 40 after the fix"
assert len(DB_META["db7"]["movement_ids"]) == 40, \\
    "DB_META['db7']['movement_ids'] should have exactly 40 entries after the fix"
'''
    if "Safety checks added by dev_history/apply_config_fix_already_applied.py" not in content:
        content += assert_block
        changes_made.append("Appended safety assert-checks at end of file")

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(content)

    print("[OK] config.py patched successfully. Changes made:")
    for c in changes_made:
        print(f"     - {c}")
    print()
    print("Now run:  python config.py")
    print("Expected output: no errors (the assert lines will run silently if correct).")


if __name__ == "__main__":
    main()
