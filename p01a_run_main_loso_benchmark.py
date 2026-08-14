"""
p01a_run_main_loso_benchmark.py — Paper 2 V2: LOSO Runner (FINAL WORKING)
===================================================================
Ridge fix: RidgeClassifierCV على 8k subsample بـ gcv_mode='svd'
  - 8k × 10k SVD: ~640 MB، ينتهي في ~30 ثانية
  - لا OOM، لا تجمد، لا init_gesdd
  - alpha محدد على نفس الـ 8k، نتائج قابلة للتكرار

الملف يغطي التجربة الرئيسية:
  - 5 methods × 3 DBs × (all subjects) LOSO
  - window=200ms, max_train=50k, kernels=10k (defaults)
  - checkpoint/resume مدمج
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import config
from data_loader import load_ninapro_stream, validate_data_path, get_n_classes
from minirocket import MiniRocket
from domain_adaptation import (
    feature_centering, coral_adaptation, tca_transform, sa_transform,
    covariance_frobenius, mmd_squared_linear,
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


def apply_adaptation(X_tr, X_te, method, **kw):
    """تطبيق طريقة الـ domain adaptation المطلوبة."""
    if method == 'raw':
        return X_tr, X_te
    if method == 'centering':
        return feature_centering(X_tr, X_te)
    if method == 'coral':
        return coral_adaptation(X_tr, X_te,
                                reg=kw.get('coral_reg', config.CORAL_REG))
    if method == 'tca':
        return tca_transform(X_tr, X_te,
                             n_components=kw.get('tca_components', config.TCA_COMPONENTS),
                             mu=kw.get('tca_mu', config.TCA_MU))
    if method == 'sa':
        return sa_transform(X_tr, X_te,
                            n_components=kw.get('sa_components', config.SA_COMPONENTS))
    raise ValueError(f"Unknown method: {method}")


# ====================================================================
# تشغيل fold واحد
# ====================================================================
def run_single_fold(train_stream, test_data, method, num_kernels, n_classes,
                    max_train_samples=50_000, per_subject_limit=5_000, **kwargs):
    t0_total = time.time()
    test_sid = test_data['subject_id']
    y_test   = test_data['y']
    X_test   = test_data['X'].astype(np.float32)

    # ── جمع بيانات التدريب ──────────────────────────────────────────
    Xl, yl = [], []
    n_subj = 0
    total  = 0
    for subj in train_stream:
        Xs = subj['X'].astype(np.float32)
        ys = subj['y']
        if len(ys) > per_subject_limit:
            Xs, ys = stratified_subsample(Xs, ys, per_subject_limit)
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
                                                 max_train_samples)

    print(f"    Train: {len(y_train):,} samples from {n_subj} subjects, "
          f"{len(np.unique(y_train))} classes", flush=True)
    print(f"    Test:  {len(y_test):,} samples, "
          f"{len(np.unique(y_test))} classes", flush=True)

    # ── MiniROCKET Transform ─────────────────────────────────────────
    t0     = time.time()
    rocket = MiniRocket(num_kernels=num_kernels,
                        random_state=config.RANDOM_SEED)
    X_tr_ppv = rocket.fit_transform(X_train)
    X_te_ppv = rocket.transform(X_test)
    t_feat   = time.time() - t0
    print(f"    PPV: train={X_tr_ppv.shape}, test={X_te_ppv.shape}",
          flush=True)

    # ── Domain shift (قبل) ───────────────────────────────────────────
    cov_before = covariance_frobenius(X_tr_ppv, X_te_ppv)
    mmd_before = mmd_squared_linear(X_tr_ppv,   X_te_ppv)

    # ── Domain Adaptation ───────────────────────────────────────────
    t0 = time.time()
    X_tr_ad, X_te_ad = apply_adaptation(X_tr_ppv, X_te_ppv,
                                         method, **kwargs)
    t_adapt = time.time() - t0

    cov_after = covariance_frobenius(X_tr_ad, X_te_ad)
    mmd_after = mmd_squared_linear(X_tr_ad,   X_te_ad)

    # ── StandardScaler ──────────────────────────────────────────────
    t0     = time.time()
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr_ad)
    X_te_sc = scaler.transform(X_te_ad)
    t_scale = time.time() - t0

    # ── Ridge (memory-safe) ─────────────────────────────────────────
    # RidgeClassifierCV على N_RIDGE=8k عينة فقط
    # gcv_mode='svd': SVD على (8k×10k) = ~640MB، آمن تماماً
    # لا init_gesdd، لا OOM، لا تجمد
    t0    = time.time()
    n_r   = min(N_RIDGE, X_tr_sc.shape[0])
    rng_r = np.random.RandomState(config.RANDOM_SEED)
    r_idx = rng_r.choice(X_tr_sc.shape[0], n_r, replace=False)

    clf = RidgeClassifierCV(
        alphas=np.logspace(-3, 6, 15),
        gcv_mode='svd',
        store_cv_results=False,
    )
    clf.fit(X_tr_sc[r_idx], y_train[r_idx])
    best_alpha = float(clf.alpha_)
    print(f"    [Ridge] n={n_r}, alpha={best_alpha:.2e}", flush=True)
    t_clf = time.time() - t0

    # ── Prediction ──────────────────────────────────────────────────
    y_pred   = clf.predict(X_te_sc)
    accuracy = float(np.mean(y_pred == y_test))

    f1s            = per_class_f1(y_test, y_pred, n_classes)
    cm             = confusion_matrix(y_test, y_pred, n_classes)
    n_test_cls     = len(np.unique(y_test))
    rand_base      = random_baseline_from_labels(y_test, n_test_cls)
    ratio          = ratio_vs_random(accuracy, rand_base)
    t_total        = time.time() - t0_total

    print(f"    Acc={accuracy:.4f}  Random={rand_base:.1f}%  "
          f"Ratio={ratio:.1f}x  [{t_total:.1f}s]", flush=True)

    result = {
        'subject_id':         int(test_sid),
        'accuracy':           round(accuracy, 6),
        'best_alpha':         best_alpha,
        'f1_per_class':       [round(f, 4) for f in f1s],
        'confusion_matrix':   cm.tolist(),
        'n_train_subjects':   n_subj,
        'n_train_samples':    int(len(y_train)),
        'n_test_samples':     int(len(y_test)),
        'n_test_classes':     n_test_cls,
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

    # تنظيف الذاكرة
    del (X_train, y_train, X_tr_ppv, X_te_ppv,
         X_tr_ad, X_te_ad, X_tr_sc, X_te_sc,
         clf, scaler)
    gc.collect()

    return result


# ====================================================================
# LOSO كامل
# ====================================================================
def run_loso_full(db_key, method, num_kernels, output_dir,
                  window_ms=200, overlap=0.5, group_classes=False,
                  resume=False, max_train_samples=50_000,
                  per_subject_limit=5_000, tag_suffix='', **kwargs):
    os.makedirs(output_dir,         exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

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

    tag = f"{db_key}_{method}"
    if tag_suffix:
        tag += f"_{tag_suffix}"

    print(f"\n{'='*72}", flush=True)
    print(f"  LOSO: {tag}", flush=True)
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

    all_accs = [v['accuracy'] for v in all_folds.values()
                if 'accuracy' in v and 'error' not in v]

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
                **kwargs,
            )
            all_folds[test_sid] = result
            if 'error' not in result:
                all_accs.append(result['accuracy'])

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
        f"  RESULT: {tag}",
        f"  Accuracy : {desc.get('mean',0):.4f} ± {desc.get('std',0):.4f}",
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
        description="Paper 2: MiniROCKET LOSO Runner (Final Working)"
    )
    p.add_argument('--db',     choices=['db7','db3','db2'], default=None)
    p.add_argument('--method', choices=config.METHODS,      default=None)
    p.add_argument('--all',    action='store_true',
                   help='شغّل جميع الـ DBs والطرق')
    p.add_argument('--num_kernels',        type=int,   default=10_000)
    p.add_argument('--window_ms',          type=int,   default=200)
    p.add_argument('--overlap',            type=float, default=0.5)
    p.add_argument('--max_train_samples',  type=int,   default=50_000)
    p.add_argument('--per_subject_limit',  type=int,   default=5_000)
    p.add_argument('--coral_reg',          type=float, default=config.CORAL_REG)
    p.add_argument('--tca_components',     type=int,   default=config.TCA_COMPONENTS)
    p.add_argument('--sa_components',      type=int,   default=config.SA_COMPONENTS)
    p.add_argument('--group_classes',      action='store_true')
    p.add_argument('--resume',             action='store_true')
    p.add_argument('--output_dir',         type=str,   default=None)
    return p.parse_args()


def main():
    args = parse_args()

    if args.all:
        databases = ['db7', 'db3', 'db2']
        methods   = config.METHODS
    else:
        databases = [args.db]     if args.db     else ['db7']
        methods   = [args.method] if args.method else config.METHODS

    output_dir = args.output_dir or config.RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    print('=' * 72, flush=True)
    print('  PAPER 2: MiniROCKET LOSO Runner', flush=True)
    print(f'  DBs={databases}  Methods={methods}', flush=True)
    print(f'  Kernels={args.num_kernels}  Window={args.window_ms}ms  '
          f'MaxTrain={args.max_train_samples:,}', flush=True)
    print(f'  Ridge: RidgeClassifierCV on {N_RIDGE} samples (gcv_mode=svd)', flush=True)
    print('=' * 72, flush=True)

    all_summaries = {}
    for db_key in databases:
        for method in methods:
            print(f"\n{'#'*72}", flush=True)
            print(f"  EXPERIMENT: {db_key} / {method}", flush=True)
            print(f"{'#'*72}", flush=True)
            try:
                result = run_loso_full(
                    db_key=db_key,
                    method=method,
                    num_kernels=args.num_kernels,
                    output_dir=output_dir,
                    window_ms=args.window_ms,
                    overlap=args.overlap,
                    group_classes=args.group_classes,
                    resume=args.resume,
                    max_train_samples=args.max_train_samples,
                    per_subject_limit=args.per_subject_limit,
                    coral_reg=args.coral_reg,
                    tca_components=args.tca_components,
                    sa_components=args.sa_components,
                )
                key = f"{db_key}_{method}"
                s   = result['summary']
                all_summaries[key] = {
                    'mean': s['mean'], 'std': s['std'],
                    'random_pct': s['avg_random_baseline_pct'],
                    'ratio': s['avg_ratio_vs_random'],
                }
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  ERROR in {db_key}/{method}: {e}", flush=True)

    # جدول ملخص نهائي
    print(f"\n{'='*72}", flush=True)
    print("  FINAL SUMMARY TABLE", flush=True)
    print(f"  {'Experiment':<25} {'Acc':>8} {'±Std':>7} "
          f"{'Random':>8} {'Ratio':>7}", flush=True)
    print(f"  {'-'*55}", flush=True)
    for key, v in all_summaries.items():
        print(f"  {key:<25} {v['mean']:>7.4f}  {v['std']:>6.4f}  "
              f"{v['random_pct']:>6.1f}%  {v['ratio']:>5.1f}x", flush=True)
    print('=' * 72, flush=True)

    # حفظ الجدول الملخص
    summary_path = os.path.join(output_dir, 'MASTER_SUMMARY.json')
    with open(summary_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\n  Master summary -> {summary_path}", flush=True)


if __name__ == '__main__':
    main()