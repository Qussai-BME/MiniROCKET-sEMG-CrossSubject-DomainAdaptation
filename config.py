"""
config.py — Paper 2 V2: GOLDEN CENTRAL CONFIGURATION (FIXED v2)
================================================================
التغييرات عن النسخة السابقة:
  - DB7:  sampling_rate 100 → 2000 Hz  (مؤكد من الأدبيات: Delsys Trigno)
  - DB2:  sampling_rate 100 → 2000 Hz  (مؤكد من الأدبيات: Delsys Trigno)
  - DB2/DB3: تمت إضافة علامة exercise_e1_only لتحميل Exercise 1 فقط
             (لتجنب تصادم الـ labels بين Exercise B وC)
  - movement_ids: مضمون يطابق n_movements بدقة (الـ global label map)
  - DEFAULT_WINDOW_MS يبقى 200ms — لكنه الآن يحسب 400 عينة (صحيح)

المراجع:
  DB2/DB3: Atzori et al. 2014, 2kHz Delsys Trigno, 50 gestures (17+23+9+rest)
  DB7:     Krasoulis et al. 2017, 2kHz Delsys Trigno, 41 gestures
           مؤكد في: Frontiers Bioeng. 2021, doi:10.3389/fbioe.2021.548357
"""
import os

# =============================================================================
# PATHS — Edit DB_PATHS to point to your local NinaPro download.
#   Option 1: Set environment variables  NINAPRO_DB7, NINAPRO_DB3, NINAPRO_DB2
#   Option 2: Edit the empty strings below to your actual paths, e.g. r"E:\NinaProDB7"
#   BASE_DIR is auto-detected as the directory containing this config.py.
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FEATURES_DIR = os.path.join(OUTPUT_DIR, "features")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLES_DIR  = os.path.join(OUTPUT_DIR, "tables")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
ABLATION_DIR = os.path.join(OUTPUT_DIR, "ablation")
LOG_DIR     = os.path.join(OUTPUT_DIR, "logs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

DB_PATHS = {
    "db7": os.environ.get("NINAPRO_DB7", r"E:\NinaProDB7"),
    "db3": os.environ.get("NINAPRO_DB3", r"E:\NinaProDB3"),
    "db2": os.environ.get("NINAPRO_DB2", r"E:\NinaProDB2"),
}

# =============================================================================
# DATABASE METADATA — مُحقَّق من الأدبيات الأصلية
# =============================================================================
DB_META = {
    # -------------------------------------------------------------------------
    # DB7 — Krasoulis et al. 2017
    # 22 subjects (intact), Delsys Trigno 2kHz, 41 movements (Exercise 1 فقط)
    # -------------------------------------------------------------------------
    "db7": {
        "n_subjects":    22,
        "n_movements":   40,
        "n_channels":    12,
        "sampling_rate": 2000,          # ← FIXED (كان 100, خطأ)
        "subject_ids":   list(range(1, 23)),
        "movement_ids":  list(range(1, 41)),   # 1..40 (was incorrectly 1..41)
        "exclude_rest":  True,
        "display_name":  "NinaPro DB7 (Mixed)",
        "exercise_ranges": {"E1": (1, 41), "all": (1, 41)},
        "exercise_e1_only": False,      # DB7 has a single exercise file
        "grouped_classes": None,
    },
    # -------------------------------------------------------------------------
    # DB3 — Atzori et al. 2014 (amputees)
    # 11 subjects, Delsys Trigno 2kHz, Exercise B only (17 movements)
    # Exercise C (23 mvt) and D (9 mvt) excluded — label collision prevention
    # -------------------------------------------------------------------------
    "db3": {
        "n_subjects":    11,
        "n_movements":   17,
        "n_channels":    12,
        "sampling_rate": 2000,          # ← مؤكد (صحيح في النسخة السابقة أيضاً)
        "subject_ids":   list(range(1, 12)),
        "movement_ids":  list(range(1, 18)),   # 1..17 → map إلى 0..16
        "exclude_rest":  True,
        "display_name":  "NinaPro DB3 (Amputees)",
        "exercise_ranges": {"E1": (1, 17)},
        "exercise_e1_only": True,       # ← مهم: نحمّل Exercise B فقط
        # تجميع 17 حركة إلى 6 مجموعات وظيفية (للـ ablation study)
        "grouped_classes": {
            1: 0, 2: 0, 3: 0, 4: 0,     # Hand open/close → Grasp
            5: 1, 6: 1,                  # Wrist flex/extend → Wrist
            7: 2, 8: 2,                  # Pronation/Supination → Rotation
            9: 3, 10: 3, 11: 3, 12: 3,  # Power/fine grasp → Precision
            13: 4, 14: 4,               # Lateral/tripod → Pinch
            15: 5, 16: 5, 17: 5,        # Thumb movements → Thumb
        },
    },
    # -------------------------------------------------------------------------
    # DB2 — Atzori et al. 2014 (intact)
    # DB2: 40 subjects, Delsys Trigno 2kHz, all three exercises (49 movements)
    # -------------------------------------------------------------------------
    "db2": {
        "n_subjects":    40,
        "n_movements":   49,
        "n_channels":    12,
        "sampling_rate": 2000,          # ← FIXED (كان 100, خطأ)
        "subject_ids":   list(range(1, 41)),
        "movement_ids":  list(range(1, 50)),   # E1: 1..17, E2: 18..40, E3: 41..49
        "exclude_rest":  True,
        "display_name":  "NinaPro DB2 (Intact)",
        "exercise_ranges": {"E1": (1, 17), "E2": (18, 40), "E3": (41, 49), "all": (1, 49)},
        "exercise_e1_only": False,      # full subject: retain E1, E2, and E3
        "grouped_classes": None,
    },
}

# =============================================================================
# DEFAULT HYPERPARAMETERS
# =============================================================================
DEFAULT_WINDOW_MS   = 200    # ms — عند 2000Hz → 400 عينة (صحيح الآن)
DEFAULT_OVERLAP     = 0.5
DEFAULT_NUM_KERNELS = 10000
DEFAULT_MAX_DILATIONS = 32

# Domain Adaptation
CORAL_REG      = 1e-3
TCA_COMPONENTS = 50
TCA_MU         = 1.0
SA_COMPONENTS  = 50

# =============================================================================
# ABLATION SWEEP VALUES
# =============================================================================
ABLATION_WINDOWS    = [100, 200, 400, 800]      # ms
ABLATION_KERNELS    = [1000, 5000, 10000]
ABLATION_CORAL_REGS = [1e-2, 1e-3, 1e-4]

# =============================================================================
# METHODS & LABELS
# =============================================================================
METHODS = ["raw", "centering", "coral", "tca", "sa"]
METHOD_LABELS = {
    "raw":       "Raw (No Adaptation)",
    "centering": "Feature Centering",
    "coral":     "CORAL",
    "tca":       "TCA",
    "sa":        "Subspace Alignment",
}
METHOD_COLORS = {
    "raw":       "#E53935",
    "centering": "#FB8C00",
    "coral":     "#43A047",
    "tca":       "#1E88E5",
    "sa":        "#8E24AA",
}

# =============================================================================
# STATISTICAL SETTINGS
# =============================================================================
RANDOM_SEED  = 42
STATS_ALPHA  = 0.05

# =============================================================================
# ADDITIONS (post-review revision) — see CHANGELOG.md
# =============================================================================

# --- Feature extractor -----------------------------------------------
# "canonical" = sktime/aeon MiniRocketMultivariate (PRIMARY)
# "custom_ppv" = this repo's original hand-written MiniRocket (RETAINED as a
#                clearly labeled comparator (no reported result uses it).
#                instruction — never silently dropped).
FEATURE_EXTRACTOR = "canonical"
FEATURE_EXTRACTOR_OPTIONS = ("canonical", "custom_ppv")
# p01a's --all-extractors flag runs both and tags results
# "<tag>__canonical" / "<tag>__custom_ppv" so neither is lost.

# --- Item 3: evaluation-paradigm labels (terminology fix) -------------------
# raw/centering use ZERO target information (inductive LOSO).
# coral/tca/sa use unlabeled target CALIBRATION windows to fit the adaptation
# transform (transductive UDA) — they must never be called "zero-shot" or
# "no target data" in text or logs.
INDUCTIVE_METHODS    = ("raw", "centering")
TRANSDUCTIVE_METHODS = ("coral", "tca", "sa")
EVALUATION_PARADIGM = {
    "raw":       "inductive_loso",
    "centering": "inductive_loso",
    "coral":     "transductive_uda",
    "tca":       "transductive_uda",
    "sa":        "transductive_uda",
}

# --- Item 4: calibration / final-test split for the transductive setting ----
# Fraction of the held-out (target) subject's REPETITIONS (not windows) used
# as unlabeled calibration data to fit coral/tca/sa. The remaining
# repetitions are the final-test set, never touched during adaptation
# fitting, for ANY method (raw/centering evaluate on the same final-test
# partition so all 5 methods remain comparable on identical test data).
CALIBRATION_FRACTION = 0.5
MIN_CALIBRATION_REPS_PER_CLASS = 1
MIN_FINAL_TEST_REPS_PER_CLASS  = 1

# --- Item 5: within-subject baseline split unit ------------------------------
# Splitting must happen on whole repetitions BEFORE windowing, never on
# individual (50%-overlapping) windows, which would leak raw samples across
# the train/test boundary.
WITHIN_SUBJECT_TEST_FRACTION = 0.2
WITHIN_SUBJECT_SPLIT_UNIT = "repetition"   # was, implicitly, "window"

# --- Item 6: methods included in INFERENTIAL statistics ---------------------
# "centering" is a numerically exact null replicate of "raw" in this
# pipeline (StandardScaler already mean-centers) and must not be treated as
# an independent condition in the Friedman/Wilcoxon/Nemenyi machinery.
# It is still computed and reported descriptively in Table 2.
INFERENTIAL_METHODS = ("raw", "coral", "tca", "sa")   # centering excluded

# --- Item 8: seeds for the main experiment -----------------------------------
SEED_LIST = [42, 123, 2024, 7, 999]

# --- Item 9: LOSO-fold terminology (73 subjects, 365 method-fold evals) -----
N_HELD_OUT_SUBJECT_FOLDS = 73          # 22 (DB7) + 40 (DB2) + 11 (DB3)
N_METHOD_FOLD_EVALUATIONS = 365        # 73 subjects x 5 methods

# --- Item 10: sensitivity grids (selected WITHOUT looking at target labels) -
CORAL_REG_GRID = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
TCA_SA_COMPONENTS_GRID = [10, 25, 50, 100, 200]

# =============================================================================
# VISUALIZATION
# =============================================================================
FIGURE_DPI    = 300
FIGURE_FORMAT = "png"
FONT_FAMILY   = "serif"
FONT_SIZE     = 12

# =============================================================================
# DB7 class grouping: 41 → 6 functional groups (للـ ablation)
# =============================================================================
DB7_GROUP_MAP = {}
for _m in range(1, 42):
    if _m <= 7:
        DB7_GROUP_MAP[_m] = 0   # Basic hand movements
    elif _m <= 14:
        DB7_GROUP_MAP[_m] = 1   # Isometric hand configurations
    elif _m <= 21:
        DB7_GROUP_MAP[_m] = 2   # Individual finger movements
    elif _m <= 28:
        DB7_GROUP_MAP[_m] = 3   # Wrist movements
    elif _m <= 35:
        DB7_GROUP_MAP[_m] = 4   # Grasp patterns
    else:
        DB7_GROUP_MAP[_m] = 5   # Precision / functional

# =============================================================================
# MOVEMENT NAMES (للتوثيق وعناوين الأشكال)
# =============================================================================
DB7_MOVEMENT_NAMES = [
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
]

# DB2 full protocol — 49 movements; E2/E3 use stable protocol IDs when
# the hand-specific names are not available in the released data.
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
] + [f"E2 movement {i}" for i in range(18, 41)] \
  + [f"E3 movement {i}" for i in range(41, 50)]

DB7_GROUP_NAMES = [
    "Basic hand", "Isometric config.", "Finger individ.",
    "Wrist", "Grasp patterns", "Precision/func.",
]
DB3_GROUP_NAMES = ["Grasp", "Wrist", "Rotation", "Precision", "Pinch", "Thumb"]

# --- Safety checks added by dev_history/apply_config_fix_already_applied.py ---
assert len(DB2_MOVEMENT_NAMES) == 50, \
    f"DB2_MOVEMENT_NAMES has {len(DB2_MOVEMENT_NAMES)} entries, expected 50 (rest + 49)"
assert len(DB7_MOVEMENT_NAMES) == 41, \
    f"DB7_MOVEMENT_NAMES has {len(DB7_MOVEMENT_NAMES)} entries, expected 41 (rest + 40)"
assert DB_META["db7"]["n_movements"] == 40, \
    "DB_META['db7']['n_movements'] should be 40 after the fix"
assert len(DB_META["db7"]["movement_ids"]) == 40, \
    "DB_META['db7']['movement_ids'] should have exactly 40 entries after the fix"
