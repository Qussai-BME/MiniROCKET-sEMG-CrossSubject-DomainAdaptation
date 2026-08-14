"""
[ARCHIVED — ALREADY APPLIED, kept for provenance only]
This reference text has already been applied to config.py's
DB2_MOVEMENT_NAMES / DB7_MOVEMENT_NAMES lists. See dev_history/README.md.
------------------------------------------------------------------------------
dev_history/movement_names_reference_notes.py
================================
Drop-in replacement for the DB2_MOVEMENT_NAMES and DB7_MOVEMENT_NAMES blocks
currently in config.py. Paste this over the existing two lists.

SOURCING / CONFIDENCE:
  DB2_MOVEMENT_NAMES (used by both DB2 and DB3 — identical Exercise B
  protocol): HIGH confidence. Cross-verified against the official NinaPro
  Exercise B movement order (M1-M17) as independently tabulated in multiple
  peer-reviewed papers citing Atzori et al. 2014.

  DB7_MOVEMENT_NAMES: HIGH confidence on the NAMES and their source
  (Exercise B [17] + Exercise C [23] = 40, per the official ninapro.hevs.ch
  DB7 page and the Krasoulis et al. 2017 Methods text, both independently
  confirming "40 movements"). NOT yet confirmed against your specific raw
  data — run tools/check_db7_labels_against_raw_data.py first (see accompanying script) and only
  use this 40-entry list if it reports 40, not 41, distinct movement labels.

  If tools/check_db7_labels_against_raw_data.py reports 41, do NOT use this list as-is: the 41st
  label is something this list does not account for, and you'll need to
  identify what it is (an extra movement, a mislabeled transition state,
  etc.) before assigning names to confusion-matrix axes.
"""

# ============================================================
# DB2 / DB3 Exercise B — 17 movements
# ============================================================
DB2_MOVEMENT_NAMES = [
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
]

# ============================================================
# DB7 — Exercise B (17) + Exercise C (23) = 40
# ⚠️ Verify with tools/check_db7_labels_against_raw_data.py before using — see module docstring.
# ============================================================
DB7_MOVEMENT_NAMES = [
    "rest",
    # --- Exercise B (17) ---
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
    # --- Exercise C (23) ---
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
]

# Fail loudly at import time if either list is ever miscounted again —
# this is exactly the class of bug that started this whole investigation.
assert len(DB2_MOVEMENT_NAMES) == 18, \
    f"DB2_MOVEMENT_NAMES has {len(DB2_MOVEMENT_NAMES)} entries, expected 18 (rest + 17)"
assert len(DB7_MOVEMENT_NAMES) == 41, \
    f"DB7_MOVEMENT_NAMES has {len(DB7_MOVEMENT_NAMES)} entries, expected 41 (rest + 40)"

if __name__ == "__main__":
    print("DB2_MOVEMENT_NAMES: OK,", len(DB2_MOVEMENT_NAMES), "entries (rest + 17)")
    print("DB7_MOVEMENT_NAMES: OK,", len(DB7_MOVEMENT_NAMES), "entries (rest + 40)")
    print()
    print("Reminder: DB7's count is well-sourced but NOT yet confirmed against")
    print("your raw data. Run tools/check_db7_labels_against_raw_data.py before relying on this list.")