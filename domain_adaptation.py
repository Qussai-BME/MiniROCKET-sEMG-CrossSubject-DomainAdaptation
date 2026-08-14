"""
domain_adaptation.py — Paper 2 V2: GOLDEN DOMAIN ADAPTATION (FIXED v2)
========================================================================
الإصلاحات عن النسخة السابقة:

  BUG-CORAL (حرج): CORAL OOM على مجموعة الاختبار الكبيرة
    قبل: X_target → float64 → (203554, 5000) × 8 bytes = 7.58 GB → OOM
    بعد:
      1. float32 طوال الدالة بالكامل (50% تخفيض في الذاكرة)
      2. تحويل مجموعة الاختبار الكبيرة بـ batching ذكي
         (كل batch = 10,000 عينة × 5000 × 4 bytes = 200 MB فقط)
      3. حساب التغاير الآمن: (X.T @ X) بدل np.cov (لا ينسخ البيانات)

  ملاحظة: مع الإصلاح الأساسي في config.py (fs=2000 بدل 100)،
  مجموعة الاختبار ستصبح ~12,000 عينة بدلاً من 203,554.
  لذا batching هو طبقة أمان إضافية فقط.

  TCA: لا تغيير (لديها max_samples=6000 بالفعل)
  SA:  لا تغيير (لديها max_pca_samples=20000 بالفعل)
"""
import numpy as np
from scipy.linalg import eigh, cholesky, solve, LinAlgError
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel, pairwise_distances
import warnings

# حجم كل batch عند تحويل مجموعة الاختبار الكبيرة
_CORAL_BATCH = 10_000   # 10k × 5000 × float32 = 200 MB/batch


# ====================================================================
# 1. Feature Centering — بدون تغيير
# ====================================================================
def feature_centering(X_source, X_target):
    mu = X_source.mean(axis=0)
    return X_source - mu, X_target - mu


# ====================================================================
# 2. CORAL — مُصلَح (float32 + batched application)
# ====================================================================
def coral_adaptation(X_source, X_target, reg=1e-3):
    """
    CORAL: CORrelation ALignment
    يوائم توزيع المصدر مع الهدف عبر محاذاة التغاير.

    الإصلاح:
      - float32 طوال: يخفض استهلاك الذاكرة 50%
      - Batched transform على X_target لمنع OOM حتى مع ملايين النوافذ
      - حساب التغاير بـ (X.T @ X) بدل np.cov لتوفير الذاكرة
    """
    # تحويل إلى float32 مباشرة (الإصلاح الأهم)
    X_source = np.asarray(X_source, dtype=np.float32)
    X_target = np.asarray(X_target, dtype=np.float32)

    if X_source.ndim == 1:
        X_source = X_source.reshape(-1, 1)
    if X_target.ndim == 1:
        X_target = X_target.reshape(-1, 1)

    n_s, F = X_source.shape
    n_t     = X_target.shape[0]

    mean_s = X_source.mean(axis=0)    # (F,) float32
    mean_t = X_target.mean(axis=0)    # (F,) float32

    Xs_c = X_source - mean_s           # in-place subtraction (float32)
    # X_target - mean_t يُنفَّذ في الـ batching أدناه (لا نحتاج Xt_c كاملاً)

    # حساب التغاير: (X.T @ X) / (n-1) — لا ينسخ البيانات الكاملة
    Cs = (Xs_c.T @ Xs_c) / max(n_s - 1, 1)   # (F, F) float32

    # نحسب Ct على batch من X_target لتوفير الذاكرة
    Ct = _cov_batched(X_target, mean_t, _CORAL_BATCH)  # (F, F) float32

    Cs_reg = Cs + reg * np.eye(F, dtype=np.float32)
    Ct_reg = Ct + reg * np.eye(F, dtype=np.float32)

    # Cholesky decomposition مع fallback
    jitter = 1e-8
    Ls, Lt = None, None
    for _ in range(20):
        try:
            Ls = cholesky(Cs_reg.astype(np.float64), lower=True).astype(np.float32)
            Lt = cholesky(Ct_reg.astype(np.float64), lower=True).astype(np.float32)
            break
        except LinAlgError:
            Cs_reg = Cs_reg + jitter * np.eye(F, dtype=np.float32)
            Ct_reg = Ct_reg + jitter * np.eye(F, dtype=np.float32)
            jitter *= 10

    if Ls is None or Lt is None:
        warnings.warn("CORAL: Cholesky failed. Falling back to Eigen.")
        for mat, name in [(Cs_reg, 'Cs'), (Ct_reg, 'Ct')]:
            ev, evec = np.linalg.eigh(mat.astype(np.float64))
            ev = np.maximum(ev, 1e-12)
            L  = (evec @ np.diag(np.sqrt(ev))).astype(np.float32)
            if name == 'Cs':
                Ls = L
            else:
                Lt = L

    # Ls_inv: (F, F)
    try:
        Ls_inv = solve(Ls.astype(np.float64),
                       np.eye(F)).astype(np.float32)
    except LinAlgError:
        Ls_inv = np.linalg.pinv(Ls.astype(np.float64)).astype(np.float32)

    # مصفوفة التحويل: W = Ls_inv @ Lt  → (F, F)
    W = Ls_inv @ Lt     # float32 @ float32 → (F, F)

    # تحويل المصدر (عادةً أصغر، لا يحتاج batching)
    Xs_aligned = (Xs_c @ W + mean_t).astype(np.float32)

    # تحويل الهدف — batched لمنع OOM
    Xt_aligned = _apply_transform_batched(X_target, mean_t, Lt, _CORAL_BATCH)

    return Xs_aligned, Xt_aligned


def _cov_batched(X, mean, batch_size):
    """
    حساب مصفوفة التغاير (F×F) بطريقة آمنة على الذاكرة.
    يعمل batch-by-batch بدون حفظ X - mean بالكامل.
    """
    n, F = X.shape
    C = np.zeros((F, F), dtype=np.float32)
    for i in range(0, n, batch_size):
        chunk = X[i:i + batch_size].astype(np.float32) - mean
        C += chunk.T @ chunk
    return C / max(n - 1, 1)


def _apply_transform_batched(X, mean, L, batch_size):
    """
    تطبيق التحويل: (X - mean) @ L + mean
    بشكل batched لتوفير الذاكرة.
    """
    n = X.shape[0]
    result = np.empty_like(X, dtype=np.float32)
    for i in range(0, n, batch_size):
        chunk = X[i:i + batch_size].astype(np.float32) - mean
        result[i:i + batch_size] = (chunk @ L) + mean
    return result


# ====================================================================
# 3. TCA — بدون تغيير (لديها max_samples=6000 بالفعل)
# ====================================================================
def tca_transform(X_source, X_target, n_components=50, mu=1.0,
                  kernel='linear', max_samples=6000, random_state=42):
    X_source = np.asarray(X_source, dtype=np.float64)
    X_target = np.asarray(X_target, dtype=np.float64)

    n_s_full = X_source.shape[0]
    n_t_full = X_target.shape[0]
    n_total  = n_s_full + n_t_full

    if n_total > max_samples:
        rng   = np.random.RandomState(random_state)
        n_s_s = min(n_s_full, max(100, int(max_samples * n_s_full / n_total)))
        n_t_s = min(n_t_full, max(50,  int(max_samples * n_t_full / n_total)))
        s_idx = rng.choice(n_s_full, n_s_s, replace=False)
        t_idx = rng.choice(n_t_full, n_t_s, replace=False)
        X_source_fit = X_source[s_idx]
        X_target_fit = X_target[t_idx]
        print(f"      [TCA] Subsampling: source {n_s_full}->{n_s_s}, "
              f"target {n_t_full}->{n_t_s} for fitting", flush=True)
    else:
        X_source_fit = X_source
        X_target_fit = X_target
        n_s_s = n_s_full
        n_t_s = n_t_full

    n_s = X_source_fit.shape[0]
    n_t = X_target_fit.shape[0]
    n   = n_s + n_t

    scaler = StandardScaler()
    X_all  = scaler.fit_transform(np.vstack([X_source_fit, X_target_fit]))

    if kernel == 'linear':
        K = X_all @ X_all.T
    elif kernel == 'rbf':
        dists    = pairwise_distances(X_all)
        gamma    = 1.0 / (2.0 * np.median(dists[dists > 0]) ** 2 + 1e-12)
        K        = rbf_kernel(X_all, gamma=gamma)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    e = np.vstack([np.ones((n_s, 1)) / n_s, -np.ones((n_t, 1)) / n_t])
    L = e @ e.T
    H = np.eye(n) - np.ones((n, n)) / n
    A = 0.5 * (x := K @ L @ K + mu * np.eye(n)) + 0.5 * x.T
    B = 0.5 * (y := K @ H @ K) + 0.5 * y.T

    try:
        eigvals, eigvecs = eigh(B, A)
    except LinAlgError:
        A += mu * 100 * np.eye(n)
        try:
            eigvals, eigvecs = eigh(B, A)
        except LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(np.linalg.pinv(A) @ B)

    idx = np.argsort(-eigvals)[:n_components]
    W   = eigvecs[:, np.sort(idx)]

    if kernel == 'linear':
        P          = X_all.T @ W
        Z_source   = scaler.transform(X_source) @ P
        Z_target   = scaler.transform(X_target) @ P
    else:
        X_full_all = scaler.transform(np.vstack([X_source, X_target]))
        K_full     = (rbf_kernel(X_full_all, X_all, gamma=gamma)
                      if kernel == 'rbf' else X_full_all @ X_all.T)
        Z_full     = K_full @ W
        Z_source   = Z_full[:n_s_full]
        Z_target   = Z_full[n_s_full:]

    return Z_source.astype(np.float32), Z_target.astype(np.float32)


# ====================================================================
# 4. SA — بدون تغيير (لديها max_pca_samples=20000 بالفعل)
# ====================================================================
def sa_transform(X_source, X_target, n_components=50, random_state=42):
    X_source = np.asarray(X_source, dtype=np.float64)
    X_target = np.asarray(X_target, dtype=np.float64)

    mean_s = X_source.mean(axis=0)
    mean_t = X_target.mean(axis=0)
    Xs_c   = X_source - mean_s
    Xt_c   = X_target - mean_t

    max_pca = 20_000
    rng     = np.random.RandomState(random_state)

    Xs_pca = (Xs_c[rng.choice(Xs_c.shape[0], max_pca, replace=False)]
              if Xs_c.shape[0] > max_pca else Xs_c)
    Xt_pca = (Xt_c[rng.choice(Xt_c.shape[0], max_pca, replace=False)]
              if Xt_c.shape[0] > max_pca else Xt_c)

    pca_s = PCA(n_components=n_components, svd_solver='randomized',
                random_state=random_state)
    pca_t = PCA(n_components=n_components, svd_solver='randomized',
                random_state=random_state)
    pca_s.fit(Xs_pca)
    pca_t.fit(Xt_pca)

    Ps = pca_s.components_.T
    Pt = pca_t.components_.T
    M  = Ps.T @ Pt

    # تأكد من توافق الأبعاد
    mean_t_sub = mean_t[:n_components] if len(mean_t) > n_components else mean_t

    Xs_aligned = (Xs_c @ Ps @ M + mean_t_sub).astype(np.float32)
    Xt_aligned = (Xt_c @ Pt   + mean_t_sub).astype(np.float32)

    return Xs_aligned, Xt_aligned


# ====================================================================
# 5. قياس انحراف المجال — بدون تغيير
# ====================================================================
def covariance_frobenius(X_source, X_target, max_samples=50_000,
                         random_state=42):
    X_source = np.asarray(X_source, dtype=np.float64)
    X_target = np.asarray(X_target, dtype=np.float64)
    if X_source.shape[0] < 3 or X_target.shape[0] < 3:
        return np.nan
    rng = np.random.RandomState(random_state)
    if X_source.shape[0] > max_samples:
        X_source = X_source[rng.choice(X_source.shape[0], max_samples, replace=False)]
    if X_target.shape[0] > max_samples:
        X_target = X_target[rng.choice(X_target.shape[0], max_samples, replace=False)]
    try:
        Xs = X_source.astype(np.float32)
        Xt = X_target.astype(np.float32)
        Xs_c = Xs - Xs.mean(axis=0)
        Xt_c = Xt - Xt.mean(axis=0)
        Cs = (Xs_c.T @ Xs_c) / max(Xs_c.shape[0] - 1, 1)
        Ct = (Xt_c.T @ Xt_c) / max(Xt_c.shape[0] - 1, 1)
        return float(np.linalg.norm(Cs - Ct, ord='fro'))
    except Exception as e:
        warnings.warn(f"covariance_frobenius failed: {e}")
        return np.nan


def mmd_squared_linear(X_source, X_target):
    X_source = np.asarray(X_source, dtype=np.float64)
    X_target = np.asarray(X_target, dtype=np.float64)
    if X_source.shape[0] == 0 or X_target.shape[0] == 0:
        return np.nan
    delta = X_source.mean(axis=0) - X_target.mean(axis=0)
    return float(delta @ delta)


def compute_domain_shift_before_after(X_source_train, X_target_train,
                                      X_source_test, X_target_test,
                                      method_func, **kwargs):
    cov_before = covariance_frobenius(X_source_train, X_target_train)
    mmd_before = mmd_squared_linear(X_source_train, X_target_train)
    Xs_al, Xt_al = method_func(X_source_train, X_target_train, **kwargs)
    cov_after  = covariance_frobenius(Xs_al, Xt_al)
    mmd_after  = mmd_squared_linear(Xs_al, Xt_al)
    return {
        'cov_before':  cov_before,
        'cov_after':   cov_after,
        'mmd_before':  mmd_before,
        'mmd_after':   mmd_after,
        'X_source_aligned': Xs_al,
        'X_target_aligned': Xt_al,
        'X_source_test_aligned': X_source_test,
        'X_target_test_aligned': X_target_test,
        'reduction_cov': ((cov_before - cov_after) / cov_before
                          if cov_before and cov_before > 0 else 0.0),
        'reduction_mmd': ((mmd_before - mmd_after) / mmd_before
                          if mmd_before and mmd_before > 0 else 0.0),
    }