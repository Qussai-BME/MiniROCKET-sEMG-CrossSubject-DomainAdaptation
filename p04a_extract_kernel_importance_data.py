"""
p04a_extract_kernel_importance_data.py — استخراج بيانات أهمية النوى (خفيف)
===========================================================================
تخفيف عن النسخة السابقة (كانت ~6.2 ساعة، الآن ~35-40 دقيقة):
  NUM_KERNELS: 10,000 -> 3,000
  MAX_TRAIN:   30,000 -> 15,000
  N_FOLDS:     5      -> 3

هذا كافٍ تماماً لغرض التحليل (اتجاه عام لأهمية النوى، ليس دقة قصوى).
كل fold يطبع تقدمه فوراً (flush=True) ليظهر فوراً في الطرفية
بشكل حي وليس بعد الانتهاء الكامل.
"""
import sys, os, gc, time, argparse
sys.path.insert(0, '.')
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import config
from data_loader import load_ninapro_stream, get_n_classes
from minirocket import MiniRocket
from domain_adaptation import (
    feature_centering, coral_adaptation, tca_transform, sa_transform,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV

N_FOLDS_SAMPLE   = 3        # ← تخفيف من 5
MAX_TRAIN        = 15_000   # ← تخفيف من 30,000
PER_SUBJ_LIMIT   = 3_000    # ← تخفيف من 5,000
NUM_KERNELS      = 3_000    # ← تخفيف من 10,000 (كافٍ لاتجاه عام)
WINDOW_MS        = 200
N_RIDGE          = 4_000    # ← تخفيف من 8,000 (يتناسب مع MAX_TRAIN الأصغر)


def apply_adaptation(X_tr, X_te, method, **kw):
    if method == 'raw':
        return X_tr, X_te
    if method == 'centering':
        return feature_centering(X_tr, X_te)
    if method == 'coral':
        return coral_adaptation(X_tr, X_te, reg=config.CORAL_REG)
    if method == 'tca':
        return tca_transform(X_tr, X_te, n_components=config.TCA_COMPONENTS,
                             mu=config.TCA_MU)
    if method == 'sa':
        return sa_transform(X_tr, X_te, n_components=config.SA_COMPONENTS)
    raise ValueError(method)


def stratified_subsample(X, y, n_max, seed=42):
    if len(y) <= n_max:
        return X, y
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    idx = []
    for c in classes:
        c_idx = np.where(y == c)[0]
        n_c = max(1, int(len(c_idx) / len(y) * n_max))
        n_c = min(n_c, len(c_idx))
        idx.extend(rng.choice(c_idx, n_c, replace=False))
    idx = np.array(idx)
    rng.shuffle(idx)
    return X[idx], y[idx]


def extract_one(db_key, method, test_sids):
    n_classes = get_n_classes(db_key, config)
    imp_rows  = []
    dilations = None

    for fold_i, test_sid in enumerate(test_sids):
        t0 = time.time()
        print(f"    [{db_key}/{method}] fold {fold_i+1}/{len(test_sids)} "
              f"(subject {test_sid}) starting...", flush=True)

        Xl, yl, total = [], [], 0
        for s in load_ninapro_stream(db_key, config, WINDOW_MS, 0.5):
            if s['subject_id'] == test_sid:
                continue
            Xs, ys = s['X'].astype(np.float32), s['y']
            if len(ys) > PER_SUBJ_LIMIT:
                Xs, ys = stratified_subsample(Xs, ys, PER_SUBJ_LIMIT)
            Xl.append(Xs); yl.append(ys)
            total += len(ys)
            if total >= MAX_TRAIN:
                break
        if not Xl:
            print(f"    [WARN] No training data for fold {fold_i+1}. Skipping.",
                  flush=True)
            continue
        X_train = np.vstack(Xl); y_train = np.concatenate(yl)
        del Xl, yl
        if len(y_train) > MAX_TRAIN:
            X_train, y_train = stratified_subsample(X_train, y_train, MAX_TRAIN)

        print(f"    [{db_key}/{method}] fold {fold_i+1}: "
              f"{len(y_train):,} train samples collected ({time.time()-t0:.1f}s)",
              flush=True)

        rocket   = MiniRocket(num_kernels=NUM_KERNELS, random_state=config.RANDOM_SEED)
        X_tr_ppv = rocket.fit_transform(X_train)
        if dilations is None:
            dilations = rocket.dilations.copy()
        print(f"    [{db_key}/{method}] fold {fold_i+1}: "
              f"MiniROCKET transform done ({time.time()-t0:.1f}s)", flush=True)

        X_te_dummy = X_tr_ppv[:min(1500, X_tr_ppv.shape[0])]
        X_tr_ad, _ = apply_adaptation(X_tr_ppv, X_te_dummy, method)

        scaler  = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_ad)

        n_r   = min(N_RIDGE, X_tr_sc.shape[0])
        rng_r = np.random.RandomState(config.RANDOM_SEED)
        r_idx = rng_r.choice(X_tr_sc.shape[0], n_r, replace=False)

        clf = RidgeClassifierCV(alphas=np.logspace(-3, 6, 10))
        clf.fit(X_tr_sc[r_idx], y_train[r_idx])

        coef = np.abs(clf.coef_)
        if coef.ndim == 2:
            coef = coef.mean(axis=0)

        # فحص حرج: TCA وSA يُسقطان البيانات على n_components (افتراضياً 50)
        # وليس على فضاء النوى الأصلي (num_kernels، مثلاً 3000). في هذه
        # الحالة |coef_| لا يقابل النوى الأصلية إطلاقاً، وتحليل "أهمية
        # النواة حسب التمدد" لا معنى له علمياً لهذه الطرق. نتخطى الحفظ
        # بدل إنتاج ملف .npz بأبعاد متعارضة (importance بطول 50 مقابل
        # dilations بطول 3000) كان سيُسبب IndexError لاحقاً في day4.
        if dilations is not None and len(coef) != len(dilations):
            print(f"    [{db_key}/{method}] fold {fold_i+1}: SKIPPED — "
                  f"coef length ({len(coef)}) != dilations length "
                  f"({len(dilations)}). This method projects to a reduced "
                  f"feature space (n_components), so per-kernel dilation "
                  f"importance does not apply. ({time.time()-t0:.1f}s)",
                  flush=True)
            del X_train, y_train, X_tr_ppv, X_tr_ad, X_tr_sc, clf, scaler
            gc.collect()
            continue

        imp_rows.append(coef.astype(np.float32))

        print(f"    [{db_key}/{method}] fold {fold_i+1}/{len(test_sids)} "
              f"DONE ({time.time()-t0:.1f}s total)", flush=True)

        del X_train, y_train, X_tr_ppv, X_tr_ad, X_tr_sc, clf, scaler
        gc.collect()

    if not imp_rows:
        return None, None
    return np.vstack(imp_rows), dilations


def main():
    global NUM_KERNELS   # يجب أن يكون أول سطر في الدالة قبل أي استخدام للاسم

    p = argparse.ArgumentParser()
    p.add_argument('--n_folds', type=int, default=N_FOLDS_SAMPLE)
    p.add_argument('--num_kernels', type=int, default=NUM_KERNELS)
    p.add_argument('--dbs', nargs='+', default=['db7', 'db3', 'db2'])
    p.add_argument('--methods', nargs='+', default=config.METHODS)
    args = p.parse_args()

    NUM_KERNELS = args.num_kernels

    os.makedirs(config.FEATURES_DIR, exist_ok=True)
    print(f"  [Config] kernels={NUM_KERNELS}, max_train={MAX_TRAIN:,}, "
          f"folds/experiment={args.n_folds}", flush=True)

    for db_key in args.dbs:
        print(f"\n  Collecting subject IDs for {db_key}...", flush=True)
        all_sids = []
        for s in load_ninapro_stream(db_key, config, WINDOW_MS, 0.5):
            all_sids.append(s['subject_id'])
        if not all_sids:
            print(f"  [WARN] No subjects for {db_key}. Skipping.", flush=True)
            continue
        print(f"  {db_key}: {len(all_sids)} subjects found.", flush=True)

        rng = np.random.RandomState(config.RANDOM_SEED)
        n_sample = min(args.n_folds, len(all_sids))
        test_sids = sorted(rng.choice(all_sids, n_sample, replace=False).tolist())

        for method in args.methods:
            tag = f"{db_key}_{method}"
            out_path = os.path.join(config.FEATURES_DIR, f"kernel_imp_{tag}.npz")
            if os.path.exists(out_path):
                print(f"  [SKIP] {out_path} already exists.", flush=True)
                continue

            # تخطٍّ مبكر (قبل أي حساب مكلف): TCA وSA يُسقطان الأبعاد إلى
            # n_components (50) بشكل ثابت دائماً، بينما dilations دائماً
            # بطول NUM_KERNELS (3000). النتيجة معروفة سلفاً بلا أي حاجة
            # لتشغيل MiniROCKET transform الكامل (5-8 دقائق/fold) فقط
            # لاكتشاف نفس النتيجة المضمونة مسبقاً في النهاية.
            if method in ('tca', 'sa'):
                n_comp = (config.TCA_COMPONENTS if method == 'tca'
                         else config.SA_COMPONENTS)
                print(f"  [SKIP-EARLY] {tag}: {method.upper()} projects to "
                      f"n_components={n_comp} (fixed) != num_kernels="
                      f"{NUM_KERNELS}. Per-kernel dilation importance never "
                      f"applies to this method -- skipping without running "
                      f"the expensive MiniROCKET extraction.", flush=True)
                continue

            print(f"\n{'='*60}", flush=True)
            print(f"  Extracting: {tag}  ({n_sample} sample folds)", flush=True)
            print(f"{'='*60}", flush=True)

            t_exp = time.time()
            imp_matrix, dilations = extract_one(db_key, method, test_sids)
            if imp_matrix is None:
                print(f"  [WARN] No data extracted for {tag}", flush=True)
                continue

            np.savez_compressed(out_path,
                               importance_matrix=imp_matrix,
                               dilations=dilations)
            print(f"  Saved: {out_path}  shape={imp_matrix.shape}  "
                  f"({time.time()-t_exp:.1f}s)", flush=True)


if __name__ == '__main__':
    main()