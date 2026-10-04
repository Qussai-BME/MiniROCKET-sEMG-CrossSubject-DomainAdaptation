"""
p01a_run_main_loso_benchmark.py — LOSO benchmark runner (canonical MiniROCKET, calibration/final-test split)
=================================================================================
هذا الملف مبني على النسخة V2 ("FINAL WORKING") مع تطبيق التعديلات
التالية (انظر CHANGELOG.md للتفاصيل الكاملة والاختبارات الاصطناعية):

  ITEM 1 (مستخرج قانوني): يُختار مستخرج الخصائص عبر
    feature_extractors.get_feature_extractor() — "canonical" (sktime
    MiniRocketMultivariate، أساسي) أو "custom_ppv" (التطبيق اليدوي
    الأصلي، مُبقى عليه كمقارن موسوم بوضوح). استخدم --all_extractors
    لتشغيل الاثنين ووسم النتائج بوضوح.

  ITEM 3 (تسمية دقيقة): raw/centering = LOSO استقرائي (inductive) —
    صفر بيانات هدف. coral/tca/sa = UDA تحويلي (transductive) — تستخدم
    نوافذ هدف غير موسومة (calibration) فقط، ولا تُوصف أبداً بأنها
    "zero-shot" أو "no target data".

  ITEM 4 (فصل calibration/final-test): تكرارات الشخص المستهدَف (وليس
    نوافذه) تُقسَّم عبر subject_split.split_by_repetition إلى:
      - calibration (غير موسومة): تُغذّي coral/tca/sa فقط لحساب
        التحويل (Ct، الفراغ الفرعي، إلخ).
      - final-test: لا تدخل أبداً في ملاءمة أي طريقة — تُستخدم فقط
        للتقييم النهائي، ولجميع الطرق الخمس على حدٍّ سواء (لضمان
        قابلية مقارنة عادلة: نفس مجموعة الاختبار لكل الطرق).

  ITEM 7 (مقاييس إضافية): balanced_accuracy وmacro_f1 يُضافان كنتائج
    أساسية بجانب accuracy (per-class F1 وconfusion matrix كانا موجودين
    أصلاً).

  ITEM 8 (seeds متعددة): seed يُمرَّر بشكل صريح عبر كل سلسلة الاستدعاء
    (rocket random_state، RNG اختيار عينات Ridge) بدل الاعتماد الضمني
    على config.RANDOM_SEED وحده. شغّل عدة seeds عبر
    p08_run_multiseed_sweep.py.

Ridge fix (من V2، بدون تغيير): RidgeClassifierCV على 8k subsample بـ
  gcv_mode='svd' — لا OOM، لا تجمد، قابل للتكرار.
"""
import sys
import os
import json
import time
import pickle
import argparse
import gc
import warnings
from datetime import datetime

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import balanced_accuracy_score, f1_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import config
from data_loader import load_ninapro_stream, validate_data_path, get_n_classes
from feature_extractors import get_feature_extractor
from subject_split import split_by_repetition
from domain_adaptation import (
    feature_centering, coral_adaptation, tca_transform, sa_transform,
    covariance_frobenius, mmd_squared_linear, BatchedStandardScaler,
)
from statistical_utils import (
    per_class_f1, confusion_matrix, descriptive_stats,
    random_baseline_from_labels, ratio_vs_random,
)

warnings.filterwarnings('ignore')

# ─── حد عينات Ridge ────────────────────────────────────────────────
# 8k × 10k مصفوفة X -> SVD آمنة تماماً (~640 MB، ~30s)
# 200 عينة/صنف لـ 40 صنف ← كافٍ لـ Ridge الخطي
N_RIDGE = 8_000


# ====================================================================
# أدوات مساعدة
# ====================================================================
def stratified_subsample(X, y, n_max, seed=42):
    """عينة طبقية بحجم n_max مع الحفاظ على توزيع الأصناف."""
    if len(y) <= n_max:
        return X, y
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    idx = []
    for c in classes:
        c_idx = np.where(y == c)[0]
        n_c   = max(1, int(len(c_idx) / len(y) * n_max))
        n_c   = min(n_c, len(c_idx))
        idx.extend(rng.choice(c_idx, n_c, replace=False))
    idx = np.array(idx)
    rng.shuffle(idx)
    return X[idx], y[idx]


def apply_adaptation(X_tr, X_te_calib, X_te_final, method, **kw):
    """
    تطبيق طريقة الـ domain adaptation المطلوبة.

    raw/centering (استقرائية) تتجاهلان X_te_calib تماماً
    ولا تريان منه أي شيء — تُقيَّمان مباشرة على X_te_final. coral/tca/sa
    (تحويلية) تُلائم تحويلها على (X_tr, X_te_calib) فقط ثم تُسقط
    X_te_final بنفس التحويل المُدرَّب، دون أي إعادة ملاءمة عليه.

    تُعيد دائماً 3 عناصر: (X_tr_transformed, X_te_final_transformed,
    n_calibration_windows_used).
    """
    if method == 'raw':
        return X_tr, X_te_final, 0
    if method == 'centering':
        # feature_centering تستخدم متوسط المصدر فقط (لا هدف إطلاقاً)
        Xtr_c, Xte_c = feature_centering(X_tr, X_te_final)
        return Xtr_c, Xte_c, 0
    if method == 'coral':
        Xtr_a, _Xcalib_a, Xfinal_a = coral_adaptation(
            X_tr, X_te_calib, X_te_final,
            reg=kw.get('coral_reg', config.CORAL_REG))
        return Xtr_a, Xfinal_a, X_te_calib.shape[0]
    if method == 'tca':
        Xtr_a, _Xcalib_a, Xfinal_a = tca_transform(
            X_tr, X_te_calib, X_te_final,
            n_components=kw.get('tca_components', config.TCA_COMPONENTS),
            mu=kw.get('tca_mu', config.TCA_MU))
        return Xtr_a, Xfinal_a, X_te_calib.shape[0]
    if method == 'sa':
        Xtr_a, _Xcalib_a, Xfinal_a = sa_transform(
            X_tr, X_te_calib, X_te_final,
            n_components=kw.get('sa_components', config.SA_COMPONENTS))
        return Xtr_a, Xfinal_a, X_te_calib.shape[0]
    raise ValueError(f"Unknown method: {method}")


# ====================================================================
# تشغيل fold واحد
# ====================================================================
def run_single_fold(train_stream, test_data, method, num_kernels, n_classes,
                    max_train_samples=50_000, per_subject_limit=5_000,
                    seed=42, feature_extractor=None,
                    calibration_fraction=None, **kwargs):
    t0_total = time.time()
    test_sid = test_data['subject_id']

    seed = config.RANDOM_SEED if seed is None else seed
    feature_extractor = feature_extractor or config.FEATURE_EXTRACTOR
    calibration_fraction = (config.CALIBRATION_FRACTION
                            if calibration_fraction is None
                            else calibration_fraction)

    is_transductive = method in config.TRANSDUCTIVE_METHODS
    paradigm = config.EVALUATION_PARADIGM.get(method, 'unknown')

    # ── تقسيم تكرارات الشخص المستهدَف ──────────────────
    # نفس التقسيم يُستخدم لكل الطرق الخمس (لضمان مجموعة اختبار موحّدة
    # قابلة للمقارنة)؛ فقط coral/tca/sa تستهلك فعلياً جزء calibration.
    y_full   = test_data['y']
    X_full   = test_data['X'].astype(np.float32)
    rep_full = test_data.get('rep_id')
    if rep_full is None:
        raise RuntimeError(
            "test_data missing 'rep_id' — data_loader.py must be the "
            "FIX-DATALOADER-01 patched version. Re-check your install."
        )

    mask_final, mask_calib, split_warnings = split_by_repetition(
        y_full, rep_full, held_out_fraction=calibration_fraction, seed=seed,
        min_reps_per_class_each_side=(
            config.MIN_FINAL_TEST_REPS_PER_CLASS
            if calibration_fraction < 0.5 else
            config.MIN_CALIBRATION_REPS_PER_CLASS
        ),
    )
    X_test_final, y_test = X_full[mask_final], y_full[mask_final]
    X_test_calib          = X_full[mask_calib]     # غير موسوم عمداً بعد هذا السطر

    if len(y_test) == 0:
        return {'subject_id': test_sid, 'accuracy': 0.0,
                'error': 'Empty final-test partition after repetition split',
                'n_train_subjects': 0, 'n_train_samples': 0,
                'n_test_samples': 0}

    # ── جمع بيانات التدريب ──────────────────────────────────────────
    Xl, yl = [], []
    n_subj = 0
    total  = 0
    for subj in train_stream:
        Xs = subj['X'].astype(np.float32)
        ys = subj['y']
        if len(ys) > per_subject_limit:
            Xs, ys = stratified_subsample(Xs, ys, per_subject_limit, seed=seed)
        Xl.append(Xs)
        yl.append(ys)
        total  += len(ys)
        n_subj += 1
        if total >= max_train_samples:
            break

    if not Xl:
        return {'subject_id': test_sid, 'accuracy': 0.0,
                'error': 'No training data',
                'n_train_subjects': 0, 'n_train_samples': 0,
                'n_test_samples': len(y_test)}

    X_train = np.vstack(Xl)
    y_train = np.concatenate(yl)
    del Xl, yl

    if len(y_train) > max_train_samples:
        X_train, y_train = stratified_subsample(X_train, y_train,
                                                 max_train_samples, seed=seed)

    print(f"    [{paradigm}] Train: {len(y_train):,} samples from {n_subj} "
          f"subjects, {len(np.unique(y_train))} classes", flush=True)
    print(f"    Test(final): {len(y_test):,} samples "
          f"({len(np.unique(y_test))} classes)  |  "
          f"Calib(unlabeled, {'USED' if is_transductive else 'unused'}): "
          f"{X_test_calib.shape[0]:,} samples", flush=True)
    if split_warnings:
        for w in split_warnings[:3]:
            print(f"    [split WARN] {w}", flush=True)

    # ── MiniROCKET Transform (مستخرج قابل للاختيار) ──────
    t0     = time.time()
    rocket = get_feature_extractor(feature_extractor,
                                   num_kernels=num_kernels,
                                   random_state=seed)
    X_tr_ppv    = rocket.fit_transform(X_train)
    X_te_ppv    = rocket.transform(X_test_final)
    X_calib_ppv = (rocket.transform(X_test_calib)
                  if (is_transductive and X_test_calib.shape[0] > 0)
                  else np.zeros((0, X_tr_ppv.shape[1]), dtype=np.float32))
    t_feat = time.time() - t0
    print(f"    [{feature_extractor}] train={X_tr_ppv.shape}, "
          f"test_final={X_te_ppv.shape}, calib={X_calib_ppv.shape}",
          flush=True)

    # MEMORY FIX: الإشارة الخام (X_train/X_test_final/X_test_calib) لم
    # تَعُد تُستخدَم بعد استخراج الخصائص — تحريرها الآن يوفّر ~640MB+
    # قبل أن تصل الذاكرة إلى ضغطها الأقصى لاحقاً في Ridge.
    n_calib_available = X_test_calib.shape[0]   # يُحفظ قبل الحذف أدناه
    del X_train, X_test_final, X_test_calib
    gc.collect()

    # ── Domain shift (قبل) — يُحسب دائماً على final، للمقارنة الموحّدة ──
    cov_before = covariance_frobenius(X_tr_ppv, X_te_ppv)
    mmd_before = mmd_squared_linear(X_tr_ppv,   X_te_ppv)

    # ── Domain Adaptation () ──────────────────────────────
    t0 = time.time()
    X_tr_ad, X_te_ad, n_calib_used = apply_adaptation(
        X_tr_ppv, X_calib_ppv, X_te_ppv, method, **kwargs)
    t_adapt = time.time() - t0

    # MEMORY FIX: حذف مراجع ما قبل المواءمة. ملاحظة: لطرق raw/centering
    # الكائن الذي تشير إليه X_tr_ad قد يكون نفس كائن X_tr_ppv بالضبط —
    # في تلك الحالة del هنا لا يُحرر شيئاً فعلياً (Python تحافظ عليه
    # بأمان لأن X_tr_ad ما زال يُشير إليه)، وهذا هو السلوك الصحيح
    # المطلوب. لطرق coral/tca/sa (كائنات جديدة تماماً)، هذا يُحرر فعلياً
    # ~1.9GB+2.3GB (train+calib+test) قبل مرحلة القياس (scaling).
    del X_tr_ppv, X_te_ppv, X_calib_ppv
    gc.collect()

    cov_after = covariance_frobenius(X_tr_ad, X_te_ad)
    mmd_after = mmd_squared_linear(X_tr_ad,   X_te_ad)

    # ── StandardScaler (نسخة آمنة للذاكرة — انظر domain_adaptation.
    # BatchedStandardScaler للسبب: sklearn's StandardScaler يُصعّد
    # داخلياً لـ float64 بصرف النظر عن dtype الدخل، فيُعيد إنتاج نفس
    # مشكلة MemoryError على مصفوفة التدريب الكاملة عند F~10,000) ──
    t0     = time.time()
    scaler = BatchedStandardScaler(batch_size=5000)
    X_tr_sc = scaler.fit_transform(X_tr_ad)
    X_te_sc = scaler.transform(X_te_ad)
    t_scale = time.time() - t0

    # MEMORY FIX: X_tr_ad/X_te_ad (قبل القياس) لم تَعُد تُستخدَم — تحرَّر
    # الآن (~1.9GB+~0.2GB) قبل الدخول إلى Ridge، وهي المرحلة الأكثر
    # حساسية للذاكرة (RidgeClassifierCV يُصعّد داخلياً لـ float64 أيضاً).
    del X_tr_ad, X_te_ad
    gc.collect()

    # ── Ridge (memory-safe) ─────────────────────────────────────────
    t0    = time.time()
    n_r   = min(N_RIDGE, X_tr_sc.shape[0])
    rng_r = np.random.RandomState(seed)      # seed المُمرَّر لا config.RANDOM_SEED الثابت
    r_idx = rng_r.choice(X_tr_sc.shape[0], n_r, replace=False)

    # MEMORY FIX الأهم: نأخذ نسخة العيّنة الفرعية (n_r × F، مثلاً
    # 8000×9996 ≈ 320MB) ثم نُحرر X_tr_sc الكاملة (1.9GB+) فوراً — بدل
    # إبقاء الاثنتين حيّتين معاً أثناء استدعاء Ridge.fit() الذي يُصعّد
    # داخلياً لـ float64 على العيّنة الفرعية (~640MB إضافية). هذا كان
    # بالضبط سبب فشل "Unable to allocate 610 MiB" رغم أن 610MB وحدها
    # صغيرة — المشكلة أن X_tr_sc الكاملة (~1.9GB) كانت لا تزال حيّة
    # بجانبها دون داعٍ.
    X_tr_sub = X_tr_sc[r_idx].copy()
    y_tr_sub = y_train[r_idx]
    del X_tr_sc
    gc.collect()

    clf = None
    last_err = None
    # عبر إصدارات sklearn المختلفة تغيّرت أسماء بعض المعاملات
    # (store_cv_values -> store_cv_results، وgcv_mode أُزيل من
    # RidgeClassifierCV في بعض الإصدارات الحديثة). نجرّب من الأكثر
    # تحديداً للأقل، بدل افتراض إصدار واحد ثابت.
    for kwargs_try in (
        dict(alphas=np.logspace(-3, 6, 15), gcv_mode='svd', store_cv_results=False),
        dict(alphas=np.logspace(-3, 6, 15), gcv_mode='svd', store_cv_values=False),
        dict(alphas=np.logspace(-3, 6, 15), store_cv_results=False),
        dict(alphas=np.logspace(-3, 6, 15), store_cv_values=False),
        dict(alphas=np.logspace(-3, 6, 15)),
    ):
        try:
            clf = RidgeClassifierCV(**kwargs_try)
            clf.fit(X_tr_sub, y_tr_sub)
            break
        except TypeError as e:
            last_err = e
            clf = None
            continue
    if clf is None:
        raise RuntimeError(
            f"RidgeClassifierCV construction failed across all known "
            f"parameter-name fallbacks (last error: {last_err}). "
            f"Check your scikit-learn version."
        )
    best_alpha = float(clf.alpha_)
    print(f"    [Ridge] n={n_r}, alpha={best_alpha:.2e}, seed={seed}", flush=True)
    t_clf = time.time() - t0
    del X_tr_sub, y_tr_sub
    gc.collect()

    # ── Prediction ──────────────────────────────────────────────────
    y_pred   = clf.predict(X_te_sc)
    accuracy = float(np.mean(y_pred == y_test))

    # مقاييس إضافية أساسية (ليس accuracy وحدها)
    bal_acc  = float(balanced_accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average='macro',
                              zero_division=0))

    f1s            = per_class_f1(y_test, y_pred, n_classes)
    cm             = confusion_matrix(y_test, y_pred, n_classes)
    n_test_cls     = len(np.unique(y_test))
    rand_base      = random_baseline_from_labels(y_test, n_test_cls)
    ratio          = ratio_vs_random(accuracy, rand_base)
    t_total        = time.time() - t0_total

    print(f"    Acc={accuracy:.4f}  BalAcc={bal_acc:.4f}  "
          f"MacroF1={macro_f1:.4f}  Random={rand_base:.1f}%  "
          f"Ratio={ratio:.1f}x  [{t_total:.1f}s]", flush=True)

    result = {
        'subject_id':         int(test_sid),
        'accuracy':           round(accuracy, 6),
        'balanced_accuracy':  round(bal_acc, 6),
        'macro_f1':           round(macro_f1, 6),
        'best_alpha':         best_alpha,
        'f1_per_class':       [round(f, 4) for f in f1s],
        'confusion_matrix':   cm.tolist(),
        'n_train_subjects':   n_subj,
        'n_train_samples':    int(len(y_train)),
        'n_test_samples':     int(len(y_test)),
        'n_test_classes':     n_test_cls,
        'n_calibration_samples_available': int(n_calib_available),
        'n_calibration_samples_used':      int(n_calib_used),
        'evaluation_paradigm': paradigm,
        'feature_extractor':   feature_extractor,
        'seed':                int(seed),
        'random_baseline_pct': round(rand_base, 2),
        'ratio_vs_random':    round(ratio, 2),
        'domain_shift': {
            'cov_before': (round(cov_before, 4)
                           if cov_before is not None and not np.isnan(cov_before)
                           else None),
            'cov_after':  (round(cov_after, 4)
                           if cov_after is not None and not np.isnan(cov_after)
                           else None),
            'mmd_before': (round(mmd_before, 6)
                           if mmd_before is not None and not np.isnan(mmd_before)
                           else None),
            'mmd_after':  (round(mmd_after, 6)
                           if mmd_after is not None and not np.isnan(mmd_after)
                           else None),
        },
        'timing': {
            'feature_extraction': round(t_feat,  2),
            'adaptation':         round(t_adapt, 2),
            'scaling':            round(t_scale, 2),
            'classification':     round(t_clf,   2),
            'total':              round(t_total, 2),
        },
    }

    # تنظيف الذاكرة (معظم المصفوفات الكبيرة حُرِّرت مبكراً أعلاه بالفعل —
    # هذا فقط لما تبقى حياً: y_train، X_te_sc، clf، scaler)
    del y_train, X_te_sc, clf, scaler
    gc.collect()

    return result


# ====================================================================
# LOSO كامل
# ====================================================================
def run_loso_full(db_key, method, num_kernels, output_dir,
                  window_ms=200, overlap=0.5, group_classes=False,
                  resume=False, max_train_samples=50_000,
                  per_subject_limit=5_000, tag_suffix='',
                  seed=42, feature_extractor=None,
                  calibration_fraction=None, max_subjects=None,
                  **kwargs):
    os.makedirs(output_dir,         exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    feature_extractor = feature_extractor or config.FEATURE_EXTRACTOR
    seed = config.RANDOM_SEED if seed is None else seed

    valid, msg = validate_data_path(db_key, config)
    if not valid:
        raise FileNotFoundError(f"Data validation failed: {msg}")

    n_classes = get_n_classes(db_key, config, group_classes)

    # جمع معرّفات الأشخاص
    subject_ids = []
    for s in load_ninapro_stream(db_key, config, window_ms, overlap,
                                  group_classes):
        subject_ids.append(s['subject_id'])
    if not subject_ids:
        raise RuntimeError(f"No subjects found for {db_key}")

    if max_subjects is not None and max_subjects < len(subject_ids):
        n_original = len(subject_ids)
        rng_subj = np.random.RandomState(seed)
        subject_ids = sorted(rng_subj.choice(
            subject_ids, size=max_subjects, replace=False).tolist())
        print(f"  [subject-subset] Using {max_subjects}/{n_original} "
              f"subjects (seed={seed}) for a faster sweep run", flush=True)

    # الوسم الآن يتضمن المستخرج وseed دائماً لمنع
    # الكتابة فوق نتائج مختلفة (canonical مقابل custom_ppv، seed مقابل آخر)
    tag = f"{db_key}_{method}__{feature_extractor}__seed{seed}"
    if tag_suffix:
        tag += f"_{tag_suffix}"

    paradigm = config.EVALUATION_PARADIGM.get(method, 'unknown')
    print(f"\n{'='*72}", flush=True)
    print(f"  LOSO: {tag}", flush=True)
    print(f"  Paradigm={paradigm}  Extractor={feature_extractor}  Seed={seed}",
          flush=True)
    print(f"  Subjects={len(subject_ids)}, Classes={n_classes}, "
          f"Window={window_ms}ms, MaxTrain={max_train_samples:,}, "
          f"Kernels={num_kernels}", flush=True)
    print(f"{'='*72}", flush=True)

    # Checkpoint
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"ckpt_{tag}.pkl")
    all_folds = {}
    if resume and os.path.exists(ckpt_path):
        try:
            with open(ckpt_path, 'rb') as f:
                all_folds = pickle.load(f)
            n_done = sum(1 for v in all_folds.values() if 'error' not in v)
            print(f"  [RESUME] {n_done}/{len(subject_ids)} folds loaded",
                  flush=True)
        except Exception as e:
            print(f"  [WARN] Checkpoint failed: {e}", flush=True)
            all_folds = {}

    all_accs     = [v['accuracy'] for v in all_folds.values()
                    if 'accuracy' in v and 'error' not in v]
    all_bal_accs = [v['balanced_accuracy'] for v in all_folds.values()
                    if 'balanced_accuracy' in v and 'error' not in v]
    all_macro_f1 = [v['macro_f1'] for v in all_folds.values()
                    if 'macro_f1' in v and 'error' not in v]

    for idx, test_sid in enumerate(subject_ids):
        if test_sid in all_folds and 'error' not in all_folds[test_sid]:
            print(f"\n  [{idx+1}/{len(subject_ids)}] "
                  f"Subject {test_sid}: SKIP (done)", flush=True)
            continue

        print(f"\n  [{idx+1}/{len(subject_ids)}] "
              f"Test Subject {test_sid}...", flush=True)

        # generator للتدريب (يستثني شخص الاختبار)
        def make_train_gen(sid=test_sid):
            for s in load_ninapro_stream(db_key, config, window_ms,
                                          overlap, group_classes):
                if s['subject_id'] != sid:
                    yield s

        # بيانات الاختبار
        test_data = None
        for s in load_ninapro_stream(db_key, config, window_ms,
                                      overlap, group_classes):
            if s['subject_id'] == test_sid:
                test_data = s
                break

        if test_data is None:
            print(f"    WARNING: Subject {test_sid} not found", flush=True)
            all_folds[test_sid] = {
                'subject_id': test_sid, 'accuracy': 0.0,
                'error': 'subject not found'
            }
            continue

        try:
            result = run_single_fold(
                make_train_gen(), test_data, method, num_kernels,
                n_classes,
                max_train_samples=max_train_samples,
                per_subject_limit=per_subject_limit,
                seed=seed,
                feature_extractor=feature_extractor,
                calibration_fraction=calibration_fraction,
                **kwargs,
            )
            all_folds[test_sid] = result
            if 'error' not in result:
                all_accs.append(result['accuracy'])
                all_bal_accs.append(result['balanced_accuracy'])
                all_macro_f1.append(result['macro_f1'])

        except Exception as e:
            import traceback
            traceback.print_exc()
            all_folds[test_sid] = {
                'subject_id': test_sid, 'accuracy': 0.0,
                'error': str(e)
            }

        # حفظ checkpoint
        try:
            with open(ckpt_path, 'wb') as f:
                pickle.dump(all_folds, f)
        except Exception as e:
            print(f"  [WARN] Checkpoint save failed: {e}", flush=True)

        del test_data
        gc.collect()

    # ── ملخص النتائج ────────────────────────────────────────────────
    if all_accs:
        desc = descriptive_stats(all_accs)
    else:
        desc = {'mean': 0.0, 'std': 0.0, 'median': 0.0,
                'min': 0.0, 'max': 0.0,
                'ci95_lower': 0.0, 'ci95_upper': 0.0}
    desc_bal_acc = descriptive_stats(all_bal_accs) if all_bal_accs else {}
    desc_macro_f1 = descriptive_stats(all_macro_f1) if all_macro_f1 else {}

    rand_vals  = [v.get('random_baseline_pct', 0)
                  for v in all_folds.values()
                  if 'random_baseline_pct' in v]
    ratio_vals = [v.get('ratio_vs_random', 0)
                  for v in all_folds.values()
                  if 'ratio_vs_random' in v]
    avg_rand   = float(np.mean(rand_vals))  if rand_vals  else 0.0
    avg_ratio  = float(np.mean(ratio_vals)) if ratio_vals else 0.0

    output = {
        'experiment': tag,
        'database':   db_key,
        'method':     method,
        'evaluation_paradigm': paradigm,
        'feature_extractor':   feature_extractor,
        'seed':                seed,
        'calibration_fraction': (config.CALIBRATION_FRACTION
                                 if calibration_fraction is None
                                 else calibration_fraction),
        'num_kernels': num_kernels,
        'window_ms':  window_ms,
        'max_train_samples': max_train_samples,
        'n_classes':  n_classes,
        'n_subjects': len(subject_ids),
        'n_folds_ok': len(all_accs),
        'timestamp':  datetime.now().isoformat(),
        'summary': {
            'mean':         round(desc.get('mean',       0.0), 4),
            'std':          round(desc.get('std',        0.0), 4),
            'median':       round(desc.get('median',     0.0), 4),
            'min':          round(desc.get('min',        0.0), 4),
            'max':          round(desc.get('max',        0.0), 4),
            'ci95_lower':   round(desc.get('ci95_lower', 0.0), 4),
            'ci95_upper':   round(desc.get('ci95_upper', 0.0), 4),
            'n':            len(all_accs),
            'avg_random_baseline_pct': round(avg_rand,  2),
            'avg_ratio_vs_random':     round(avg_ratio, 2),
            # — بجانب accuracy، ليست بديلاً عنها
            'balanced_accuracy_mean': round(desc_bal_acc.get('mean', 0.0), 4),
            'balanced_accuracy_std':  round(desc_bal_acc.get('std', 0.0), 4),
            'macro_f1_mean':          round(desc_macro_f1.get('mean', 0.0), 4),
            'macro_f1_std':           round(desc_macro_f1.get('std', 0.0), 4),
        },
        'per_subject': [
            all_folds[sid] for sid in sorted(all_folds.keys())
        ],
    }

    out_path = os.path.join(output_dir, f"results_{tag}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=str)

    # طباعة ملخص نظيف
    lines = [
        f"\n{'='*72}",
        f"  RESULT: {tag}   [{paradigm}]",
        f"  Accuracy         : {desc.get('mean',0):.4f} ± {desc.get('std',0):.4f}",
        f"  Balanced Accuracy: {desc_bal_acc.get('mean',0):.4f} ± {desc_bal_acc.get('std',0):.4f}",
        f"  Macro-F1         : {desc_macro_f1.get('mean',0):.4f} ± {desc_macro_f1.get('std',0):.4f}",
        f"  Median   : {desc.get('median',0):.4f}",
        f"  Range    : [{desc.get('min',0):.4f}, {desc.get('max',0):.4f}]",
        f"  95% CI   : [{desc.get('ci95_lower',0):.4f}, {desc.get('ci95_upper',0):.4f}]",
        f"  Random   : {avg_rand:.1f}%   Ratio: {avg_ratio:.1f}x",
        f"  Folds    : {len(all_accs)}/{len(subject_ids)} OK",
        f"  Saved    : {out_path}",
        f"{'='*72}",
    ]
    for line in lines:
        print(line, flush=True)

    # حفظ ملف ملخص نصي
    txt_path = os.path.join(output_dir, f"summary_{tag}.txt")
    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    return output


# ====================================================================
# CLI
# ====================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="MiniROCKET LOSO benchmark runner"
    )
    p.add_argument('--db',     choices=['db7','db3','db2'], default=None)
    p.add_argument('--method', choices=config.METHODS,      default=None)
    p.add_argument('--all',    action='store_true',
                   help='شغّل جميع الـ DBs والطرق')
    p.add_argument('--extractor', choices=config.FEATURE_EXTRACTOR_OPTIONS,
                   default=config.FEATURE_EXTRACTOR,
                   help='مستخرج الخصائص (افتراضي: canonical)')
    p.add_argument('--all_extractors', action='store_true',
                   help='شغّل canonical وcustom_ppv كلاهما، '
                        'موسومين بوضوح، بدل اختيار واحد')
    p.add_argument('--seed', type=int, default=config.RANDOM_SEED,
                   help='للتشغيل متعدد الـ seeds استخدم '
                        'p08_run_multiseed_sweep.py')
    p.add_argument('--calibration_fraction', type=float,
                   default=config.CALIBRATION_FRACTION,
                   help='نسبة تكرارات الهدف المخصصة لـ '
                        'calibration (غير موسومة)')
    p.add_argument('--num_kernels',        type=int,   default=10_000)
    p.add_argument('--window_ms',          type=int,   default=200)
    p.add_argument('--overlap',            type=float, default=0.5)
    p.add_argument('--max_train_samples',  type=int,   default=50_000)
    p.add_argument('--per_subject_limit',  type=int,   default=5_000)
    p.add_argument('--coral_reg',          type=float, default=config.CORAL_REG)
    p.add_argument('--tca_components',     type=int,   default=config.TCA_COMPONENTS)
    p.add_argument('--sa_components',      type=int,   default=config.SA_COMPONENTS)
    p.add_argument('--fit_mode', choices=['full_train', 'sampled_train'],
                   default='full_train')
    p.add_argument('--rocket_n_jobs', type=int, default=1)
    p.add_argument('--group_classes',      action='store_true')
    p.add_argument('--resume',             action='store_true')
    p.add_argument('--output_dir',         type=str,   default=None)
    return p.parse_args()


def main():
    args = parse_args()
    from optimized_pipeline import run_loso_shared

    if args.all:
        databases = ['db7', 'db3', 'db2']
        methods   = config.METHODS
    else:
        databases = [args.db]     if args.db     else ['db7']
        methods   = [args.method] if args.method else config.METHODS

    extractors = (list(config.FEATURE_EXTRACTOR_OPTIONS)
                 if args.all_extractors else [args.extractor])

    output_dir = args.output_dir or config.RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    print('=' * 72, flush=True)
    print('  MiniROCKET LOSO Runner', flush=True)
    print(f'  DBs={databases}  Methods={methods}  Extractors={extractors}  '
          f'Seed={args.seed}', flush=True)
    print(f'  Kernels={args.num_kernels}  Window={args.window_ms}ms  '
          f'MaxTrain={args.max_train_samples:,}  '
          f'CalibFraction={args.calibration_fraction}', flush=True)
    print(f'  Ridge: RidgeClassifierCV on {N_RIDGE} samples (gcv_mode=svd)', flush=True)
    print('=' * 72, flush=True)

    all_summaries = {}
    for extractor in extractors:
        for db_key in databases:
            print(f"\n{'#'*72}", flush=True)
            print(f"  SHARED-FEATURE EXPERIMENT: {db_key} / {extractor} / seed={args.seed}", flush=True)
            print(f"{'#'*72}", flush=True)
            try:
                results = run_loso_shared(
                    db_key=db_key, methods=methods, num_kernels=args.num_kernels,
                    output_dir=output_dir, window_ms=args.window_ms, overlap=args.overlap,
                    group_classes=args.group_classes, resume=args.resume,
                    max_train_samples=args.max_train_samples,
                    per_subject_limit=args.per_subject_limit, seed=args.seed,
                    feature_extractor=extractor,
                    calibration_fraction=args.calibration_fraction,
                    coral_reg=args.coral_reg, tca_components=args.tca_components,
                    sa_components=args.sa_components, fit_mode=args.fit_mode,
                    rocket_n_jobs=args.rocket_n_jobs)
                for method, result in results.items():
                    key = f"{db_key}_{method}_{extractor}_seed{args.seed}"
                    s = result['summary']
                    all_summaries[key] = {
                        'mean': s.get('mean', 0), 'std': s.get('std', 0),
                        'balanced_accuracy_mean': s.get('balanced_accuracy_mean', 0),
                        'macro_f1_mean': s.get('macro_f1_mean', 0),
                    }
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  ERROR in {db_key}/{extractor}: {e}", flush=True)

    # جدول ملخص نهائي
    print(f"\n{'='*72}", flush=True)
    print("  FINAL SUMMARY TABLE", flush=True)
    print(f"  {'Experiment':<38} {'Acc':>7} {'BalAcc':>7} {'MacroF1':>8} "
          f"{'Ratio':>7}", flush=True)
    print(f"  {'-'*70}", flush=True)
    for key, v in all_summaries.items():
        print(f"  {key:<38} {v['mean']:>6.4f} {v['balanced_accuracy_mean']:>7.4f} "
              f"{v['macro_f1_mean']:>8.4f} {v.get('ratio', 0.0):>5.1f}x", flush=True)
    print('=' * 72, flush=True)

    # حفظ الجدول الملخص
    summary_path = os.path.join(
        output_dir, f"MASTER_SUMMARY_seed{args.seed}.json")
    with open(summary_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\n  Master summary -> {summary_path}", flush=True)


if __name__ == '__main__':
    main()
