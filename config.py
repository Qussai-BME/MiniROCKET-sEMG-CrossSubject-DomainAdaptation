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
    "db7": os.environ.get("NINAPRO_DB7", r""),
    "db3": os.environ.get("NINAPRO_DB3", r""),
    "db2": os.environ.get("NINAPRO_DB2", r""),
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
    # 40 subjects, Delsys Trigno 2kHz, Exercise B only (17 movements)
    # -------------------------------------------------------------------------
    "db2": {
        "n_subjects":    40,
        "n_movements":   17,
        "n_channels":    12,
        "sampling_rate": 2000,          # ← FIXED (كان 100, خطأ)
        "subject_ids":   list(range(1, 41)),
        "movement_ids":  list(range(1, 18)),   # 1..17 → map إلى 0..16
        "exclude_rest":  True,
        "display_name":  "NinaPro DB2 (Intact)",
        "exercise_ranges": {"E1": (1, 17)},
        "exercise_e1_only": True,       # ← مهم: نحمّل Exercise B فقط
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

# DB2 / DB3 Exercise B — 17 movements
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

DB7_GROUP_NAMES = [
    "Basic hand", "Isometric config.", "Finger individ.",
    "Wrist", "Grasp patterns", "Precision/func.",
]
DB3_GROUP_NAMES = ["Grasp", "Wrist", "Rotation", "Precision", "Pinch", "Thumb"]

# --- Safety checks added by dev_history/apply_config_fix_already_applied.py ---
assert len(DB2_MOVEMENT_NAMES) == 18, \
    f"DB2_MOVEMENT_NAMES has {len(DB2_MOVEMENT_NAMES)} entries, expected 18 (rest + 17)"
assert len(DB7_MOVEMENT_NAMES) == 41, \
    f"DB7_MOVEMENT_NAMES has {len(DB7_MOVEMENT_NAMES)} entries, expected 41 (rest + 40)"
assert DB_META["db7"]["n_movements"] == 40, \
    "DB_META['db7']['n_movements'] should be 40 after the fix"
assert len(DB_META["db7"]["movement_ids"]) == 40, \
    "DB_META['db7']['movement_ids'] should have exactly 40 entries after the fix"
