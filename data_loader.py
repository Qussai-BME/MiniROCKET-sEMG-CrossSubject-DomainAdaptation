"""
data_loader.py — Paper 2 V2: GOLDEN STREAMING LOADER (FIXED v2)
================================================================
الإصلاحات عن النسخة السابقة:

  BUG-1 (حرج): Label Alignment
    قبل:  كل شخص يُرقَّم بشكل مستقل (per-subject remap) ->
          label 3 عند شخص A ≠ label 3 عند شخص B إذا كان لديهم حركات مختلفة
    بعد:  global label map ثابت من DB_META["movement_ids"] ->
          label 3 دائماً = نفس الحركة لجميع الأشخاص

  BUG-2 (حرج لـ DB2/DB3): Exercise File Collision
    قبل:  تُحمَّل جميع ملفات التمرينات (B+C+D) -> labels 1-17 من Exercise B
          تتصادم مع labels 1-23 من Exercise C في نفس الـ label space
    بعد:  _find_subject_files_filtered() تُحمّل Exercise 1 (B) فقط
          لـ DB2/DB3 عند تفعيل exercise_e1_only=True في DB_META

  BUG-3 (تحسين): Sampling Rate
    الـ sampling_rate الصحيح موجود الآن في config.py (2000 Hz للكل)
    -> win_samples = 200ms × 2000/1000 = 400 عينة (صحيح)
    -> test windows per subject ≈ 12,000 بدلاً من 203,554

  لا يوجد تغيير في:
    - معمارية الـ streaming generator
    - _load_mat_robust / _extract_ninapro_data
    - create_windows_fast / assign_labels_to_windows
    - validate_data_path / load_ninapro_db / get_n_classes
"""
import os
import re
import gc
import glob
import warnings
import numpy as np
import scipy.io

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    warnings.warn("h5py not installed. Will not read MATLAB v7.3 files.", ImportWarning)

# ========================================================================
# 1. دوال تحميل الملفات (ROBUST I/O) — بدون تغيير
# ========================================================================

def _load_mat_robust(filepath):
    try:
        data = scipy.io.loadmat(filepath)
        return data, 'scipy'
    except NotImplementedError:
        if HAS_H5PY:
            try:
                data = h5py.File(filepath, 'r')
                return data, 'h5py'
            except Exception as e:
                warnings.warn(f"h5py failed to load {os.path.basename(filepath)}: {e}")
                return None, None
        else:
            warnings.warn(f"h5py not available for {os.path.basename(filepath)} (v7.3 file).")
            return None, None
    except Exception as e:
        warnings.warn(f"scipy failed to load {os.path.basename(filepath)}: {e}")
        return None, None


def _find_key(data, candidates):
    if isinstance(data, dict):
        for k in candidates:
            if k in data:
                return k
    elif HAS_H5PY and isinstance(data, h5py.File):
        for k in candidates:
            if k in data:
                return k
    return None


def _extract_ninapro_data(data, loader_type):
    emg_keys   = ['emg', 'EMG', 'data']
    label_keys = ['restimulus', 'stimulus']
    fs_keys    = ['sampling_frequency', 'fs']

    emg_key   = _find_key(data, emg_keys)
    label_key = _find_key(data, label_keys)
    fs_key    = _find_key(data, fs_keys)

    if emg_key is None or label_key is None:
        available = list(data.keys()) if isinstance(data, dict) else 'h5py File'
        warnings.warn(f"Missing EMG or stimulus key. Available: {available}")
        return None, None, None

    try:
        if loader_type == 'h5py':
            emg    = np.array(data[emg_key],   dtype=np.float32)
            labels = np.array(data[label_key]).squeeze().astype(np.int32)
            fs     = int(np.array(data[fs_key]).item()) if fs_key else 2000
        else:
            emg    = np.array(data[emg_key],   dtype=np.float32)
            labels = np.array(data[label_key]).squeeze().astype(np.int32)
            fs     = int(data[fs_key].item()) if fs_key else 2000

        if emg.ndim == 2 and emg.shape[0] < emg.shape[1]:
            emg = emg.T
        elif emg.ndim == 1:
            emg = emg.reshape(-1, 1)

        return emg, labels, fs

    except Exception as e:
        warnings.warn(f"Error extracting data: {e}")
        return None, None, None


# ========================================================================
# 2. FIX: Exercise-aware file finder
# ========================================================================

def _is_exercise1_file(filename):
    """
    هل هذا الملف ينتمي إلى Exercise 1 (B)؟
    يدعم أنماط التسمية الشائعة في NinaPro DB2/DB3:
      - S1_E1_A1.mat  -> نعم
      - S01_E1_A1.mat -> نعم
      - S1_B1.mat     -> نعم  (بعض النسخ)
      - S1_E2_A1.mat  -> لا
      - S1_E3_A1.mat  -> لا
      - S1_C1.mat     -> لا
      - S1_D1.mat     -> لا
    """
    fn = os.path.basename(filename).lower()

    # علامات صريحة على Exercise ≥ 2 -> رفض
    reject_patterns = ['_e2_', '_e2.', 'e2_a', '_e3_', '_e3.', 'e3_a',
                       '_c1', 'c1.mat', '_d1', 'd1.mat']
    for pat in reject_patterns:
        if pat in fn:
            return False

    # علامات صريحة على Exercise 1 -> قبول
    accept_patterns = ['_e1_', '_e1.', 'e1_a', '_b1', 'b1.mat']
    for pat in accept_patterns:
        if pat in fn:
            return True

    # لا توجد علامة exercise صريحة -> قبول (ملف واحد لهذا الشخص)
    return True


def _find_subject_files(db_path, db_key=None, meta=None):
    """
    البحث عن ملفات .mat وتجميعها حسب رقم الشخص.
    إذا كان exercise_e1_only=True في meta (لـ DB2/DB3)،
    يتم تحميل Exercise 1 فقط لمنع تصادم الـ labels.
    """
    data_path  = os.path.abspath(db_path)
    all_mats   = sorted(glob.glob(os.path.join(data_path, "**", "*.mat"),
                                  recursive=True))
    if not all_mats:
        all_mats = sorted(glob.glob(os.path.join(data_path, "*.mat")))

    e1_only = (meta is not None and meta.get("exercise_e1_only", False))
    if e1_only:
        before = len(all_mats)
        all_mats = [f for f in all_mats if _is_exercise1_file(f)]
        after = len(all_mats)
        if before != after:
            print(f"  [Loader] Exercise filter (E1 only): "
                  f"{before} -> {after} files for {db_key}", flush=True)
        if after == 0:
            warnings.warn(
                f"[Loader] No Exercise-1 files found in {data_path}! "
                "Check filename patterns. Loading all files as fallback."
            )
            all_mats = sorted(glob.glob(os.path.join(data_path, "**", "*.mat"),
                                        recursive=True))

    subject_files = {}
    for mat_path in all_mats:
        stem   = os.path.basename(mat_path)
        parent = os.path.basename(os.path.dirname(mat_path))

        subj_id = None
        m = re.search(r'[Ss](\d+)', stem)
        if m:
            subj_id = int(m.group(1))
        else:
            m = re.search(r'[Ss]ubject[ _]?(\d+)', parent, re.IGNORECASE)
            if m:
                subj_id = int(m.group(1))
            else:
                m = re.search(r'(\d+)', parent)
                if m:
                    subj_id = int(m.group(1))

        if subj_id is not None:
            subject_files.setdefault(subj_id, []).append(mat_path)

    return subject_files


# ========================================================================
# 3. دوال النوافذ — بدون تغيير
# ========================================================================

def create_windows_fast(signal, win_samples, overlap):
    if signal.ndim != 2:
        raise ValueError(f"Signal must be 2D, got {signal.shape}")
    N, C  = signal.shape
    step  = max(1, int(win_samples * (1.0 - overlap)))
    n_win = (N - win_samples) // step + 1
    if n_win <= 0:
        return np.empty((0, C, win_samples), dtype=np.float32)
    strides = (signal.strides[0] * step,
               signal.strides[1],
               signal.strides[0])
    windows = np.lib.stride_tricks.as_strided(
        signal,
        shape=(n_win, C, win_samples),
        strides=strides
    )
    return np.ascontiguousarray(windows, dtype=np.float32)


def assign_labels_to_windows(windows, labels, overlap, win_samples):
    if windows.shape[0] == 0:
        return np.array([], dtype=np.int32)
    step = max(1, int(win_samples * (1.0 - overlap)))
    mids = (np.arange(windows.shape[0]) * step + win_samples // 2).astype(np.int32)
    mids = np.clip(mids, 0, len(labels) - 1)
    return labels[mids]


# ========================================================================
# 4. دوال التصفية والتحقق — بدون تغيير جوهري
# ========================================================================

def _extract_trials(emg, stim, rep, min_len=100):
    if rep is None:
        rep = np.ones_like(stim)
    trials = []
    cur_s, cur_r, start = 0, 0, 0
    for i in range(len(stim)):
        s = int(stim[i])
        r = int(rep[i])
        if i > 0 and (s != cur_s or r != cur_r):
            length = i - start
            if cur_s > 0 and length >= min_len:
                trials.append({
                    "emg":        emg[start:i].copy(),
                    "stimulus":   cur_s,
                    "repetition": cur_r,
                })
            cur_s, cur_r, start = s, r, i
    length = len(stim) - start
    if cur_s > 0 and length >= min_len:
        trials.append({
            "emg":        emg[start:].copy(),
            "stimulus":   cur_s,
            "repetition": cur_r,
        })
    return trials


def validate_data_path(db_key, cfg):
    path = cfg.DB_PATHS.get(db_key)
    if path is None:
        return False, f"Database key '{db_key}' not found in DB_PATHS"
    if not os.path.exists(path):
        return False, f"Path does not exist: {path}"
    mat_files = glob.glob(os.path.join(path, "**", "*.mat"), recursive=True)
    if not mat_files:
        mat_files = glob.glob(os.path.join(path, "*.mat"))
    if not mat_files:
        return False, f"No .mat files found in {path}"
    return True, f"OK: {len(mat_files)} .mat files found in {path}"


# ========================================================================
# 5. MAIN GENERATOR — القلب الذهبي (مُصلَح)
# ========================================================================

def load_ninapro_stream(db_key, cfg, window_ms, overlap,
                        group_classes=False, min_trial_len=100,
                        exercise_filter=None):
    """
    مولد (Generator) يحمّل NinaPro بشكل متدفق، شخصاً في كل مرة.

    الإصلاحات:
      1. _find_subject_files الآن يستقبل db_key و meta -> Exercise 1 only
      2. Global label map من DB_META["movement_ids"] بدل per-subject remap
         -> label 3 = نفس الحركة دائماً لجميع الأشخاص في نفس الـ fold
      3. النوافذ الخارجة عن global label space تُرشح تلقائياً مع تحذير

    Yields dict per subject:
      subject_id, X (n_win, n_ch, win_samples), y (n_win,),
      n_classes, label_map, sampling_rate
    """
    db_path = cfg.DB_PATHS[db_key]
    meta    = cfg.DB_META[db_key]
    fs      = meta["sampling_rate"]          # 2000 Hz (مُصحَّح في config.py)
    n_ch    = meta["n_channels"]

    # حجم النافذة الصحيح: 200ms × 2000/1000 = 400 عينة
    win_samples = max(4, int(fs * window_ms / 1000))

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database path not found: {db_path}")

    # ---- FIX-2: Exercise-aware file discovery ----
    subject_files = _find_subject_files(db_path, db_key=db_key, meta=meta)
    if not subject_files:
        raise FileNotFoundError(f"No .mat files found in {db_path}")

    valid_ids = set(meta["subject_ids"])
    subject_files = {k: v for k, v in subject_files.items() if k in valid_ids}
    if not subject_files:
        raise RuntimeError(f"No valid subjects found in {db_path}.")

    # ---- FIX-1: Global label map (ثابت لجميع الأشخاص) ----
    movement_ids    = meta["movement_ids"]          # e.g. [1..41] or [1..17]
    global_lmap     = {int(m): i for i, m in enumerate(movement_ids)}
    global_lmap_set = set(global_lmap.keys())
    n_classes_global = len(movement_ids)

    for sid in sorted(subject_files.keys()):
        all_windows, all_labels = [], []

        for fpath in subject_files[sid]:
            mat, ltype = _load_mat_robust(fpath)
            if mat is None:
                continue

            emg, stim, fs_file = _extract_ninapro_data(mat, ltype)
            if emg is None:
                continue

            # ضبط عدد القنوات
            if emg.shape[1] > n_ch:
                emg = emg[:, :n_ch]
            elif emg.shape[1] < n_ch:
                pad = np.zeros((emg.shape[0], n_ch - emg.shape[1]), dtype=np.float32)
                emg = np.hstack([emg, pad])

            # محاولة استخراج متغير التكرار
            rep = None
            for key in mat.keys():
                if 'rep' in key.lower():
                    rep = (np.array(mat[key]).ravel()
                           if ltype == 'h5py'
                           else np.array(mat[key]).squeeze())
                    break

            trials = _extract_trials(emg, stim, rep, min_trial_len)

            for tr in trials:
                wins = create_windows_fast(tr["emg"], win_samples, overlap)
                if wins.shape[0] == 0:
                    continue

                win_lbls = assign_labels_to_windows(
                    wins,
                    np.full(tr["emg"].shape[0], tr["stimulus"], dtype=np.int32),
                    overlap, win_samples
                )

                # تطبيق فلتر التمرين الاختياري (للـ ablation)
                if exercise_filter is not None:
                    mask = _apply_exercise_filter(win_lbls, exercise_filter)
                    if mask.sum() > 0:
                        wins, win_lbls = wins[mask], win_lbls[mask]
                    else:
                        continue

                # تجميع الأصناف (لـ ablation)
                if group_classes and meta.get("grouped_classes") is not None:
                    gmap      = meta["grouped_classes"]
                    new_lbls  = np.array([gmap.get(int(l), -1) for l in win_lbls],
                                         dtype=np.int32)
                    valid_grp = new_lbls >= 0
                    if valid_grp.sum() > 0:
                        wins, win_lbls = wins[valid_grp], new_lbls[valid_grp]
                    else:
                        continue

                all_windows.append(wins)
                all_labels.append(win_lbls)

            del emg, stim, mat
            gc.collect()

        if not all_windows:
            continue

        X = np.concatenate(all_windows, axis=0).astype(np.float32)
        y = np.concatenate(all_labels).astype(np.int32)

        # استبعاد الراحة (rest = 0)
        if meta.get("exclude_rest", True):
            rest_mask = y > 0
            if rest_mask.sum() == 0:
                del X, y, all_windows, all_labels
                gc.collect()
                continue
            X, y = X[rest_mask], y[rest_mask]

        if len(y) == 0:
            continue

        # ---- FIX-1: Global label map ----
        # تصفية أي label خارج نطاق movement_ids (مثلاً بقايا Exercise C)
        valid_mask = np.isin(y, list(global_lmap_set))
        n_filtered = (~valid_mask).sum()
        if n_filtered > 0:
            warnings.warn(
                f"Subject {sid} ({db_key}): Filtered {n_filtered} windows "
                f"with labels outside global map {movement_ids[0]}..{movement_ids[-1]}. "
                "هذا يشير إلى أن ملفات Exercise غير E1 وصلت بشكل غير متوقع."
            )
        if valid_mask.sum() == 0:
            del X, y, all_windows, all_labels
            gc.collect()
            continue
        X, y = X[valid_mask], y[valid_mask]

        # الـ global label map: {1:0, 2:1, ..., 41:40} أو {1:0,...,17:16}
        # إذا كان group_classes مفعّلاً، الـ y بالفعل أُعيد تعيينه أعلاه
        if not (group_classes and meta.get("grouped_classes") is not None):
            y_remapped = np.array([global_lmap[int(l)] for l in y], dtype=np.int32)
            n_cls = n_classes_global
            final_lmap = global_lmap
        else:
            # بعد group_classes: y بالفعل 0..K-1
            y_remapped = y
            n_cls = len(set(meta["grouped_classes"].values()))
            final_lmap = {v: v for v in range(n_cls)}

        yield {
            "subject_id":   sid,
            "X":            X,
            "y":            y_remapped,
            "n_classes":    n_cls,
            "label_map":    final_lmap,
            "sampling_rate": fs,
        }

        del X, y, all_windows, all_labels
        gc.collect()


def _apply_exercise_filter(labels, ex_filter):
    """فلتر التمرين (للـ ablation فقط — عند الحاجة لـ E1+E2)."""
    if ex_filter in ('all', None):
        return labels > 0
    elif ex_filter == 'E1':
        return (labels >= 1) & (labels <= 17)
    elif ex_filter == 'E2':
        return (labels >= 18) & (labels <= 40)
    elif ex_filter == 'E1+E2':
        return (labels >= 1) & (labels <= 40)
    else:
        warnings.warn(f"Unknown exercise filter: {ex_filter}. Keeping all.")
        return np.ones(len(labels), dtype=bool)


# ========================================================================
# 6. الدالة التوافقية (للـ ablation runners)
# ========================================================================

def load_ninapro_db(db_key, cfg, window_ms=200, overlap=0.5,
                    group_classes=False, min_trial_len=100):
    """تحمّل جميع الأشخاص في dict (للسكربتات القديمة). تستهلك ذاكرة أكثر."""
    data = {}
    for subj in load_ninapro_stream(db_key, cfg, window_ms, overlap,
                                    group_classes, min_trial_len,
                                    exercise_filter=None):
        sid = subj["subject_id"]
        data[sid] = {
            "X":          subj["X"],
            "y":          subj["y"],
            "subject_id": sid,
            "label_map":  subj["label_map"],
            "n_classes":  subj["n_classes"],
        }
    return data


def get_n_classes(db_key, cfg, group_classes=False):
    meta = cfg.DB_META[db_key]
    if group_classes and meta.get("grouped_classes") is not None:
        return len(set(meta["grouped_classes"].values()))
    return meta["n_movements"]