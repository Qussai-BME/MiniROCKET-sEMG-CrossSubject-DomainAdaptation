"""
p10_run_sensitivity_analysis.py -- descriptive sensitivity sweep (feature-reuse version)
=======================================================================================
Sweeps the CORAL regularization (lambda) and the TCA / SA subspace dimension on one database
(default DB7, seed 42) and reports accuracy, balanced accuracy and macro-F1 for each grid point.

MiniROCKET features do not depend on lambda or on the number of components, so features are
extracted once per held-out subject (fold) and reused for every grid point. Each grid point is then
applied to the same matrices with `apply_adaptation()` from p01a_run_main_loso_benchmark.py,
i.e. exactly the function used by the main benchmark. This reduces extraction cost from
N_subjects x 15 to N_subjects x 1.

Resume works at subject level: a subject whose whole grid is complete is skipped; an incomplete
subject is recomputed from scratch.

The values used in the main experiment are the defaults in config.py and are NOT selected from this
sweep. No target labels are used in any grid point.

Usage
    python p10_run_sensitivity_analysis.py --db db7 --extractor canonical --resume
    python p10_run_sensitivity_analysis.py --db db7 --max_subjects 2      # quick test first
"""
import sys, os, json, argparse, time, gc, pickle, warnings
from datetime import datetime

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import config
from data_loader import load_ninapro_stream, validate_data_path, get_n_classes
from feature_extractors import get_feature_extractor
from subject_split import split_by_repetition
from domain_adaptation import BatchedStandardScaler
from statistical_utils import (per_class_f1, confusion_matrix,
                               descriptive_stats, random_baseline_from_labels,
                               ratio_vs_random)
from optimized_pipeline import atomic_pickle, stratified_subsample
from p01a_run_main_loso_benchmark import apply_adaptation
from sklearn.linear_model import RidgeClassifierCV

warnings.filterwarnings('ignore')
N_RIDGE = 8_000


def build_grid():
    """كل نقاط شبكة الحساسية الخمس عشرة (افتراضياً) كقائمة
    (method, param_name, kwarg_name, value, grid_key)."""
    grid = []
    for v in config.CORAL_REG_GRID:
        grid.append(('coral', 'coral_reg', 'coral_reg', v, f"coral_reg={v}"))
    for v in config.TCA_SA_COMPONENTS_GRID:
        grid.append(('tca', 'tca_components', 'tca_components', v,
                     f"tca_components={v}"))
    for v in config.TCA_SA_COMPONENTS_GRID:
        grid.append(('sa', 'sa_components', 'sa_components', v,
                     f"sa_components={v}"))
    return grid


def fit_predict_one_point(X_tr_ppv, X_calib_ppv, X_te_ppv, y_train, y_test,
                          n_classes, method, kwarg_name, value, seed):
    """يُطبَّق DA + scaling + Ridge على خصائص مُستخرَجة مسبقاً لنقطة شبكة
    واحدة. مطابق تماماً لذيل run_single_fold (بعد استخراج الخصائص)."""
    X_tr_ad, X_te_ad, n_calib_used = apply_adaptation(
        X_tr_ppv, X_calib_ppv, X_te_ppv, method, **{kwarg_name: value})

    scaler = BatchedStandardScaler(batch_size=5000)
    X_tr_sc = scaler.fit_transform(X_tr_ad)
    X_te_sc = scaler.transform(X_te_ad)
    del X_tr_ad, X_te_ad
    gc.collect()

    n_r = min(N_RIDGE, X_tr_sc.shape[0])
    rng_r = np.random.RandomState(seed)
    r_idx = rng_r.choice(X_tr_sc.shape[0], n_r, replace=False)
    X_tr_sub = X_tr_sc[r_idx].copy()
    y_tr_sub = y_train[r_idx]
    del X_tr_sc
    gc.collect()

    clf = None
    last_err = None
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
        raise RuntimeError(f"RidgeClassifierCV failed: {last_err}")
    best_alpha = float(clf.alpha_)
    del X_tr_sub, y_tr_sub
    gc.collect()

    y_pred = clf.predict(X_te_sc)
    accuracy = float(np.mean(y_pred == y_test))
    bal_acc  = float(balanced_accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average='macro', zero_division=0))
    f1s = per_class_f1(y_test, y_pred, n_classes)
    cm  = confusion_matrix(y_test, y_pred, n_classes)
    rand_base = random_baseline_from_labels(y_test, len(np.unique(y_test)))

    del X_te_sc, clf, scaler
    gc.collect()

    return {
        'accuracy': round(accuracy, 6),
        'balanced_accuracy': round(bal_acc, 6),
        'macro_f1': round(macro_f1, 6),
        'best_alpha': best_alpha,
        'f1_per_class': [round(f, 4) for f in f1s],
        'confusion_matrix': cm.tolist(),
        'n_calibration_samples_used': int(n_calib_used),
        'random_baseline_pct': round(rand_base, 2),
        'ratio_vs_random': round(ratio_vs_random(accuracy, rand_base), 2),
    }


def run_subject(db_key, test_sid, window_ms, overlap, group_classes,
                num_kernels, seed, feature_extractor, calibration_fraction,
                max_train_samples, per_subject_limit, n_classes, grid):
    """يُنتج dict لشخص واحد يغطي كل نقاط الشبكة، باستخراج واحد فقط."""
    test_data = None
    train_subjs = []
    for s in load_ninapro_stream(db_key, config, window_ms, overlap,
                                 group_classes):
        if s['subject_id'] == test_sid:
            test_data = s
        else:
            train_subjs.append(s)
    if test_data is None:
        return {'error': f'subject {test_sid} not found'}

    y_full   = test_data['y']
    X_full   = test_data['X'].astype(np.float32)
    rep_full = test_data.get('rep_id')
    if rep_full is None:
        raise RuntimeError("test_data missing 'rep_id' — need FIX-DATALOADER-01 loader")

    mask_final, mask_calib, _warn = split_by_repetition(
        y_full, rep_full, held_out_fraction=calibration_fraction, seed=seed,
        min_reps_per_class_each_side=(
            config.MIN_FINAL_TEST_REPS_PER_CLASS
            if calibration_fraction < 0.5 else
            config.MIN_CALIBRATION_REPS_PER_CLASS),
    )
    X_test_final, y_test = X_full[mask_final], y_full[mask_final]
    X_test_calib          = X_full[mask_calib]
    if len(y_test) == 0:
        return {'error': 'empty final-test partition after repetition split'}

    Xl, yl = [], []
    total = 0
    for subj in train_subjs:
        Xs, ys = subj['X'].astype(np.float32), subj['y']
        if len(ys) > per_subject_limit:
            Xs, ys = stratified_subsample(Xs, ys, per_subject_limit, seed)
        Xl.append(Xs); yl.append(ys); total += len(ys)
        if total >= max_train_samples:
            break
    if not Xl:
        return {'error': 'no training data'}
    X_train = np.vstack(Xl); y_train = np.concatenate(yl)
    del Xl, yl
    if len(y_train) > max_train_samples:
        X_train, y_train = stratified_subsample(X_train, y_train,
                                                 max_train_samples, seed)

    # ── الاستخراج مرة واحدة فقط لهذا الشخص ─────────────────────────
    t0 = time.time()
    rocket = get_feature_extractor(feature_extractor, num_kernels=num_kernels,
                                   random_state=seed)
    X_tr_ppv    = rocket.fit_transform(X_train)
    X_te_ppv    = rocket.transform(X_test_final)
    X_calib_ppv = (rocket.transform(X_test_calib)
                  if X_test_calib.shape[0] > 0
                  else np.zeros((0, X_tr_ppv.shape[1]), dtype=np.float32))
    t_extract = time.time() - t0
    del X_train, X_test_final, X_test_calib
    gc.collect()
    print(f"    [extract once] {t_extract:.1f}s  train={X_tr_ppv.shape} "
          f"final={X_te_ppv.shape} calib={X_calib_ppv.shape}", flush=True)

    out = {'_extract_seconds': round(t_extract, 2), '_n_train_subjects': len(train_subjs)}
    for method, param_name, kwarg_name, value, key in grid:
        t0 = time.time()
        try:
            r = fit_predict_one_point(X_tr_ppv, X_calib_ppv, X_te_ppv,
                                      y_train, y_test, n_classes,
                                      method, kwarg_name, value, seed)
            r['seconds'] = round(time.time() - t0, 2)
            out[key] = r
            print(f"      {key:22s} acc={r['accuracy']:.4f} "
                  f"bal_acc={r['balanced_accuracy']:.4f} "
                  f"macro_f1={r['macro_f1']:.4f}  ({r['seconds']:.1f}s)",
                  flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            out[key] = {'error': str(e)}

    del X_tr_ppv, X_te_ppv, X_calib_ppv, y_train, y_test
    gc.collect()
    return out


def main():
    p = argparse.ArgumentParser(description="feature-reuse sensitivity sweep")
    p.add_argument('--db', choices=['db7', 'db3', 'db2'], default='db7')
    p.add_argument('--extractor', choices=config.FEATURE_EXTRACTOR_OPTIONS,
                   default=config.FEATURE_EXTRACTOR)
    p.add_argument('--num_kernels', type=int, default=10_000)
    p.add_argument('--seed', type=int, default=config.RANDOM_SEED)
    p.add_argument('--max_subjects', type=int, default=None)
    p.add_argument('--window_ms', type=int, default=200)
    p.add_argument('--overlap', type=float, default=0.5)
    p.add_argument('--calibration_fraction', type=float,
                   default=config.CALIBRATION_FRACTION)
    p.add_argument('--max_train_samples', type=int, default=50_000)
    p.add_argument('--per_subject_limit', type=int, default=5_000)
    p.add_argument('--output_dir', type=str, default=None)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()

    output_dir = args.output_dir or config.RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    valid, msg = validate_data_path(args.db, config)
    if not valid:
        raise FileNotFoundError(f"Data validation failed: {msg}")
    n_classes = get_n_classes(args.db, config, group_classes=False)

    subject_ids = sorted({s['subject_id'] for s in load_ninapro_stream(
        args.db, config, args.window_ms, args.overlap, False)})
    if args.max_subjects is not None and args.max_subjects < len(subject_ids):
        rng = np.random.RandomState(args.seed)
        subject_ids = sorted(rng.choice(subject_ids, args.max_subjects,
                                        replace=False).tolist())
        print(f"[subject-subset] using {len(subject_ids)} subjects "
              f"(seed={args.seed})", flush=True)

    grid = build_grid()
    print(f"Grid: {len(grid)} points "
          f"({len(config.CORAL_REG_GRID)} coral_reg + "
          f"{len(config.TCA_SA_COMPONENTS_GRID)} tca_components + "
          f"{len(config.TCA_SA_COMPONENTS_GRID)} sa_components) "
          f"x {len(subject_ids)} subjects", flush=True)

    tag = f"{args.db}_sensitivity_reuse__{args.extractor}__seed{args.seed}"
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"ckpt_{tag}.pkl")
    all_subjects = {}
    if args.resume and os.path.exists(ckpt_path):
        with open(ckpt_path, 'rb') as f:
            all_subjects = pickle.load(f)
        n_done = sum(1 for v in all_subjects.values()
                    if 'error' not in v and len(v) >= len(grid) + 2)
        print(f"[RESUME] {n_done}/{len(subject_ids)} subjects already complete",
              flush=True)

    expected_keys = len(grid)
    for idx, sid in enumerate(subject_ids):
        existing = all_subjects.get(sid)
        n_ok = (sum(1 for k, v in existing.items()
                    if k in {g[4] for g in grid} and 'error' not in v)
               if existing and 'error' not in existing else 0)
        if existing and n_ok >= expected_keys:
            print(f"\n[{idx+1}/{len(subject_ids)}] Subject {sid}: SKIP (complete)",
                  flush=True)
            continue

        print(f"\n[{idx+1}/{len(subject_ids)}] Subject {sid}...", flush=True)
        t0 = time.time()
        try:
            result = run_subject(
                args.db, sid, args.window_ms, args.overlap, False,
                args.num_kernels, args.seed, args.extractor,
                args.calibration_fraction, args.max_train_samples,
                args.per_subject_limit, n_classes, grid)
        except Exception as e:
            import traceback; traceback.print_exc()
            result = {'error': str(e)}
        result['subject_id'] = int(sid)
        all_subjects[sid] = result
        atomic_pickle(all_subjects, ckpt_path)
        print(f"  [{idx+1}/{len(subject_ids)}] done in {time.time()-t0:.1f}s "
              f"(checkpoint saved)", flush=True)
        gc.collect()

    # ── تجميع النتائج بنفس شكل مخرجات p10 (متوافق مع أدوات لاحقة) ──────
    rows = []
    for method, param_name, kwarg_name, value, key in grid:
        vals_acc, vals_bal, vals_f1 = [], [], []
        for sid, sres in all_subjects.items():
            r = sres.get(key)
            if r and 'error' not in r:
                vals_acc.append(r['accuracy'])
                vals_bal.append(r['balanced_accuracy'])
                vals_f1.append(r['macro_f1'])
        if vals_acc:
            rows.append({
                'param': param_name, 'value': value,
                'accuracy_mean': round(float(np.mean(vals_acc)), 4),
                'accuracy_std':  round(float(np.std(vals_acc)), 4),
                'balanced_accuracy_mean': round(float(np.mean(vals_bal)), 4),
                'macro_f1_mean': round(float(np.mean(vals_f1)), 4),
                'n_folds': len(vals_acc),
            })

    out = {
        'database': args.db, 'feature_extractor': args.extractor,
        'seed': args.seed, 'num_kernels': args.num_kernels,
        'n_subjects_requested': len(subject_ids),
        'method': 'feature_reuse_v1',
        'note': ("Descriptive sensitivity sweep only . "
                "Features extracted once per subject and reused across "
                "all grid points — same coral_adaptation/tca_transform/"
                "sa_transform code as p01a, unmodified. The value used "
                "in the primary Table 2 experiment is fixed in config.py "
                "and was NOT selected from this sweep's results."),
        'timestamp': datetime.now().isoformat(),
        'rows': rows,
    }
    out_path = os.path.join(output_dir, f"sensitivity_analysis_{args.db}_feature_reuse.json")
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"\n{'='*72}\nDONE — {out_path}\n{'='*72}")
    for r in rows:
        print(f"  {r['param']:<16} {str(r['value']):>8} "
              f"acc={r['accuracy_mean']:.4f} bal_acc={r['balanced_accuracy_mean']:.4f} "
              f"macro_f1={r['macro_f1_mean']:.4f}  (n={r['n_folds']})")


if __name__ == '__main__':
    main()