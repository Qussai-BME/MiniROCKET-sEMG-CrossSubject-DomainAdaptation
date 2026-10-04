"""
domain_adaptation.py — CORAL implementation (corrected; see CHANGELOG.md)
========================================================================
إصلاحات CORAL (انظر CHANGELOG.md):

  FIX-CORAL-01: الهدف كان يُعاد تحويله بعد CORAL (إعادة تلوين إضافية
    غير مبررة بمصفوفة Cholesky الخاصة بتغايره هو). CORAL الصحيح يُحوّل
    المصدر فقط؛ الهدف يبقى دون أي تحويل خطي.

  FIX-CORAL-02 (اكتُشف أثناء كتابة اختبار FIX-CORAL-01 الاصطناعي):
    مصفوفة التبييض/إعادة التلوين كانت تُحسب كـ Ls^{-1} @ Lt بدل
    Ls^{-T} @ Lt^T — وهو خطأ في اتجاه المنقول يعني أن CORAL لم يكن
    يُوائم تغاير المصدر مع تغاير الهدف فعليًا (تحقّق عددي: خطأ Frobenius
    ≈285 قبل الإصلاح مقابل ≈0.35 بعده على بيانات اختبارية).

  التحقق: tools/audit_coral_synthetic_test.py — 4 حالات اصطناعية
    غاوسية بأبعاد وتشوهات مختلفة، كلها ناجحة بعد الإصلاح
    (تخفيض مسافة Frobenius بين تغايرَي المصدر والهدف بنسبة ~99.9-100%).

  NOTE: هذان الإصلاحان يعنيان أن نتائج CORAL الأصلية في المخطوطة
  (Table 2, 6-8) كانت مبنية على تطبيق خاطئ رياضيًا لـ CORAL، وليس
  فقط على خاصية بنيوية في فضاء PPV كما افتُرض. يجب إعادة تشغيل CORAL
  بالكامل على NinaPro قبل أي استنتاج حول آلية فشله (انظر خطة العمل).

========================================================================
إصلاحات الذاكرة (من النسخة السابقة، لا تزال سارية):

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
import os

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
def coral_adaptation(X_source, X_target_calib, X_target_final=None, reg=1e-3):
    """
    CORAL: CORrelation ALignment
    يوائم توزيع المصدر مع الهدف عبر محاذاة التغاير.

    FIX-DA-ITEM4: X_target_calib هو الجزء غير الموسوم من الهدف الذي
    يُستخدم لحساب mean_t/Ct (وبالتالي W). إن مُرِّر X_target_final (جزء
    اختبار منفصل تماماً)، يُعاد كما هو دون أي تحويل (لأن CORAL لا يُحوّل
    الهدف إطلاقاً بعد FIX-CORAL-01 — انظر أدناه) ضمن ثلاثي القيم
    المُعادة؛ هذا يوفر واجهة استدعاء موحّدة مع tca_transform/sa_transform
    التي تحتاج فعلاً لإعادة إسقاط الاختبار النهائي.

    الإصلاح:
      - float32 طوال: يخفض استهلاك الذاكرة 50%
      - Batched transform على X_target لمنع OOM حتى مع ملايين النوافذ
      - حساب التغاير بـ (X.T @ X) بدل np.cov لتوفير الذاكرة
    """
    # تحويل إلى float32 مباشرة (الإصلاح الأهم)
    X_source = np.asarray(X_source, dtype=np.float32)
    X_target_calib = np.asarray(X_target_calib, dtype=np.float32)

    if X_source.ndim == 1:
        X_source = X_source.reshape(-1, 1)
    if X_target_calib.ndim == 1:
        X_target_calib = X_target_calib.reshape(-1, 1)

    n_s, F = X_source.shape
    n_t     = X_target_calib.shape[0]

    mean_s = X_source.mean(axis=0)    # (F,) float32
    mean_t = X_target_calib.mean(axis=0)    # (F,) float32

    Xs_c = X_source - mean_s           # in-place subtraction (float32)
    del X_source   # لم يعد يُستخدَم بعد هذه النقطة؛ تحريره يوفر ~نفس حجم Xs_c
    # X_target_calib - mean_t يُنفَّذ في الـ batching أدناه (لا نحتاج Xt_c كاملاً)

    # حساب التغاير: (X.T @ X) / (n-1) — لا ينسخ البيانات الكاملة
    Cs = (Xs_c.T @ Xs_c) / max(n_s - 1, 1)   # (F, F) float32

    # نحسب Ct على batch من X_target_calib فقط لتوفير الذاكرة (ولضمان عدم
    # تسريب أي معلومة من X_target_final إلى W)
    Ct = _cov_batched(X_target_calib, mean_t, _CORAL_BATCH)  # (F, F) float32

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

    # FIX-CORAL-02 (بق ثانٍ، اكتُشف أثناء كتابة اختبار FIX-CORAL-01):
    # لبيانات على هيئة صفوف (n×F، كما في كل هذا المشروع)، تبييض المصدر
    # يتطلب الضرب بـ Ls^{-T} (مقلوب-منقول Cholesky) لا Ls^{-1}:
    #   cov(Xs_c @ Ls^{-T}) = Ls^{-1} · (Ls Ls^T) · Ls^{-T} = I   ✓ صحيح
    #   cov(Xs_c @ Ls^{-1})  ≠ I  بشكل عام (Ls ليست متماثلة)      ✗ خطأ
    # وبالمثل، إعادة التلوين نحو تغاير الهدف تتطلب الضرب بـ Lt^T لا Lt.
    # النسخة السابقة استخدمت Ls^{-1} @ Lt مباشرة (بدون منقول)، وهو ما
    # يعني أن CORAL لم يكن يُوائم فعليًا تغاير المصدر مع تغاير الهدف على
    # الإطلاق — تحقّق عددي: خطأ Frobenius بين cov(المصدر المحوَّل) وCt
    # كان ≈285 مع الصيغة القديمة، وانخفض إلى ≈0.35 مع الصيغة الصحيحة (على
    # بيانات اصطناعية اختبارية). هذا الخطأ مستقل عن FIX-CORAL-01 أعلاه
    # ويُصحَّح هنا معًا لأن كليهما في نفس السطر تقريبًا من الحساب.
    try:
        Ls_invT = solve(Ls.astype(np.float64).T,
                        np.eye(F)).astype(np.float32)
    except LinAlgError:
        Ls_invT = np.linalg.pinv(Ls.astype(np.float64).T).astype(np.float32)

    # مصفوفة التحويل الصحيحة: W = Ls^{-T} @ Lt^T  → (F, F)
    W = Ls_invT @ Lt.T     # float32 @ float32 → (F, F)

    # تحويل المصدر (عادةً أصغر، لا يحتاج batching)
    Xs_aligned = (Xs_c @ W + mean_t).astype(np.float32)

    # FIX-CORAL-01 (إصلاح ذو أولوية قصوى — انظر تدقيق المشرف):
    # في CORAL الأصلي (Sun, Feng & Saenko, 2016) يُعاد "تبييض ثم تلوين"
    # المصدر فقط نحو تغاير الهدف؛ الهدف نفسه لا يُحوَّل إطلاقًا — يُستخدم
    # فقط لحساب Ct وmean_t. النسخة السابقة من هذه الدالة كانت تُمرر
    # X_target عبر _apply_transform_batched(..., Lt, ...)، أي "تعيد تلوين"
    # الهدف بمصفوفة Cholesky الخاصة *بتغايره هو نفسه* — وهذا يقارب تربيع
    # تغاير الهدف (يضخّمه بدل تركه سليمًا)، وهو بالضبط ما رصده التدقيق:
    # تحويل إضافي غير مبرر يُطبَّق على الهدف. الإصلاح: الهدف يُعاد كما هو
    # دون أي تحويل خطي.
    Xt_aligned = np.array(X_target_calib, dtype=np.float32, copy=True)

    if X_target_final is None:
        return Xs_aligned, Xt_aligned      # سلوك قديم، متوافق للخلف

    # واجهة موحّدة مع tca_transform/sa_transform؛ CORAL لا يُحوّل الهدف
    # إطلاقاً (كلا الجزأين، calib وfinal)، لذا الإعادة هنا مجرد نسخ.
    Xt_final_aligned = np.array(X_target_final, dtype=np.float32, copy=True)
    return Xs_aligned, Xt_aligned, Xt_final_aligned


class BatchedStandardScaler:
    """
    بديل آمن للذاكرة عن sklearn.preprocessing.StandardScaler، لنفس الغرض
    بالضبط (تصفير المتوسط، تطبيع التباين لكل خاصية)، لكن دون أي تحويل
    float64 على المصفوفة الكاملة (n×F).

    السبب: sklearn's StandardScaler يُراكم mean_/var_ داخلياً بـ float64
    دائماً (عبر _incremental_mean_and_var) بصرف النظر عن dtype الدخل —
    هذا اختيار داخلي لدقة الحساب التراكمي، وليس شيئاً يمكن للمستدعي
    تعطيله عبر إعطاء float32 فقط. عند F~10,000 هذا يُعيد إنتاج نفس
    مشكلة MemoryError (49,978×9,996×8 بايت = 3.72GB) حتى لو كانت كل
    مصفوفاتنا الخاصة float32 من البداية.

    الحل هنا: المتوسط والتباين (mean_, std_) متجهان بطول F فقط (~80KB
    حتى بـ float64 عند F=10,000) — لا مشكلة إطلاقاً في حسابهما بدقة
    عالية. المُشكلة فقط في تطبيق (X - mean)/std على مصفوفة n×F كاملة؛
    هذا يتم هنا batch-by-batch، بحيث لا تتجاوز الذاكرة الإضافية اللحظية
    حجم دفعة واحدة (batch_size صفاً) بدل المصفوفة كاملة، وبقاء الناتج
    float32 طوال الوقت.

    النتائج العددية مطابقة عملياً لـ sklearn.StandardScaler (نفس
    الصيغة: (X - mean) / std، حيث std = sqrt(var) وvar بـ ddof=0).
    """

    def __init__(self, batch_size=5000, eps=1e-8):
        self.batch_size = batch_size
        self.eps = eps
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        n, F = X.shape
        # التراكم بدقة float64 آمن هنا لأن mean/var متجهات (F,) فقط،
        # وليست مصفوفة (n, F) كاملة.
        mean = np.zeros(F, dtype=np.float64)
        for i in range(0, n, self.batch_size):
            mean += X[i:i + self.batch_size].astype(np.float64).sum(axis=0)
        mean /= n

        var = np.zeros(F, dtype=np.float64)
        for i in range(0, n, self.batch_size):
            chunk = X[i:i + self.batch_size].astype(np.float64) - mean
            var += (chunk * chunk).sum(axis=0)
        var /= n

        std = np.sqrt(var)
        std[std < self.eps] = 1.0   # نفس معالجة sklearn للأعمدة شبه-الثابتة

        self.mean_ = mean.astype(np.float32)
        self.std_  = std.astype(np.float32)
        return self

    def transform(self, X):
        if self.mean_ is None:
            raise RuntimeError("BatchedStandardScaler must be fitted first.")
        n = X.shape[0]
        out = np.empty(X.shape, dtype=np.float32)
        for i in range(0, n, self.batch_size):
            end = min(i + self.batch_size, n)
            out[i:end] = (X[i:end].astype(np.float32) - self.mean_) / self.std_
        return out

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


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
    تطبيق التحويل: (X - mean) @ L + mean، بشكل batched لتوفير الذاكرة.

    [مُهمَل بعد FIX-CORAL-01] لم تَعُد coral_adaptation تستدعي هذه
    الدالة على الهدف (target) — الهدف لم يعد يُحوَّل إطلاقًا في CORAL،
    اتساقًا مع الخوارزمية الأصلية. أُبقيت هنا فقط لأنها قد تُستخدم
    لاحقًا إن احتجنا لتحويل مصدر ضخم بنفس أسلوب الـ batching.
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
def tca_transform(X_source, X_target_calib, X_target_final=None,
                  n_components=50, mu=1.0,
                  kernel='linear', max_samples=6000, random_state=42):
    """
    FIX-DA-ITEM4: X_target_calib هو الجزء غير الموسوم فقط من الهدف الذي
    يُستخدم لملاءمة التحويل (المصفوفة P). إن مُرِّر X_target_final (جزء
    الاختبار النهائي المنفصل تماماً)، يُسقَط عليه نفس P المُدرَّبة على
    (source, target_calib) فقط، دون أي إعادة ملاءمة — فيُعاد Z ثلاثي
    (Z_source, Z_target_calib, Z_target_final). إن لم يُمرَّر (السلوك
    القديم، متوافق للخلف)، يُعاد زوج (Z_source, Z_target_calib) كما كان.
    """
    # FIX (نفس فئة مشكلة MemoryError في covariance_frobenius): لا نُحوّل
    # المصفوفات الكاملة (قد تكون عشرات الآلاف × ~10k عمود) إلى float64.
    # float32 كافٍ تماماً لـ StandardScaler.transform ولإسقاط P لاحقاً؛
    # الدقة العددية العالية (float64) تُستخدم فقط لاحقاً على المصفوفات
    # الصغيرة المُصغَّرة (n ≤ max_samples) في حساب kernel/eigh.
    # Optional sensitivity-only cap; primary runs keep max_samples=6000.
    max_samples = int(os.environ.get('TCA_MAX_SAMPLES', max_samples))
    X_source = np.asarray(X_source, dtype=np.float32)
    X_target_calib = np.asarray(X_target_calib, dtype=np.float32)

    n_s_full = X_source.shape[0]
    n_t_full = X_target_calib.shape[0]
    n_total  = n_s_full + n_t_full

    if n_total > max_samples:
        rng   = np.random.RandomState(random_state)
        n_s_s = min(n_s_full, max(100, int(max_samples * n_s_full / n_total)))
        n_t_s = min(n_t_full, max(50,  int(max_samples * n_t_full / n_total)))
        s_idx = rng.choice(n_s_full, n_s_s, replace=False)
        t_idx = rng.choice(n_t_full, n_t_s, replace=False)
        X_source_fit = X_source[s_idx]
        X_target_fit = X_target_calib[t_idx]
        print(f"      [TCA] Subsampling: source {n_s_full}->{n_s_s}, "
              f"target(calib) {n_t_full}->{n_t_s} for fitting", flush=True)
    else:
        X_source_fit = X_source
        X_target_fit = X_target_calib
        n_s_s = n_s_full
        n_t_s = n_t_full

    n_s = X_source_fit.shape[0]
    n_t = X_target_fit.shape[0]
    n   = n_s + n_t

    scaler = StandardScaler()
    # X_all صغيرة (n ≤ max_samples) — آمن تماماً أن تكون float64 هنا
    # للاستقرار العددي في eigh/kernel أدناه.
    X_all  = scaler.fit_transform(
        np.vstack([X_source_fit, X_target_fit]).astype(np.float64))
    # FIX حرِج: scaler المُدرَّب على float64 يُبقي mean_/scale_ بـ
    # float64؛ استدعاء scaler.transform() لاحقاً على المصفوفات
    # float32 *الكاملة* (X_source، X_target_calib، X_target_final) كان
    # سيُصعّد الناتج تلقائياً إلى float64 عبر قواعد numpy للترقية —
    # فيُعيد إنتاج نفس مشكلة نفاد الذاكرة على مصفوفة كبيرة، بعد سطر
    # واحد فقط من الإصلاح أعلاه. تحويل mean_/scale_ إلى float32 صراحةً
    # يضمن بقاء ناتج transform() على المصفوفات الكبيرة float32.
    scaler.mean_  = scaler.mean_.astype(np.float32)
    scaler.scale_ = scaler.scale_.astype(np.float32)
    scaler.var_   = scaler.var_.astype(np.float32)

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
        Z_target   = scaler.transform(X_target_calib) @ P
    else:
        X_full_all = scaler.transform(np.vstack([X_source, X_target_calib]))
        K_full     = (rbf_kernel(X_full_all, X_all, gamma=gamma)
                      if kernel == 'rbf' else X_full_all @ X_all.T)
        Z_full     = K_full @ W
        Z_source   = Z_full[:n_s_full]
        Z_target   = Z_full[n_s_full:]

    Z_source = Z_source.astype(np.float32)
    Z_target = Z_target.astype(np.float32)

    if X_target_final is None:
        return Z_source, Z_target      # سلوك قديم، متوافق للخلف

    # FIX-DA-ITEM4: إسقاط الاختبار النهائي بنفس P/scaler المُدرَّبة على
    # (source, target_calib) فقط — لا إعادة ملاءمة، لا تسريب.
    X_target_final = np.asarray(X_target_final, dtype=np.float32)
    if kernel == 'linear':
        Z_target_final = scaler.transform(X_target_final) @ P
    else:
        X_final_scaled = scaler.transform(X_target_final)
        K_final = (rbf_kernel(X_final_scaled, X_all, gamma=gamma)
                  if kernel == 'rbf' else X_final_scaled @ X_all.T)
        Z_target_final = K_final @ W
    return Z_source, Z_target, Z_target_final.astype(np.float32)


# ====================================================================
# 4. SA — بدون تغيير (لديها max_pca_samples=20000 بالفعل)
# ====================================================================
def sa_transform(X_source, X_target_calib, X_target_final=None,
                 n_components=50, random_state=42):
    """
    FIX-DA-ITEM4: mean_t وPt يُشتقّان من X_target_calib فقط. إن مُرِّر
    X_target_final، يُحاذى بنفس mean_t/Pt (بدون إعادة تدريب PCA)، ويُعاد
    ثلاثي (Xs_aligned, Xt_calib_aligned, Xt_final_aligned). خلاف ذلك،
    زوج (Xs_aligned, Xt_calib_aligned) كما في السلوك القديم.
    """
    # FIX (نفس فئة مشكلة MemoryError في covariance_frobenius/TCA): لا
    # نُحوّل المصفوفات الكاملة إلى float64 — PCA بـ randomized SVD يعمل
    # جيداً تماماً على float32، وXs_c/Xt_c بحجم (n×F) هي بالضبط ما كان
    # يُسبب نفاد الذاكرة (49,978×9,996×8 بايت ≈ 3.7GB بـ float64، مقابل
    # ~1.9GB بـ float32).
    X_source = np.asarray(X_source, dtype=np.float32)
    X_target_calib = np.asarray(X_target_calib, dtype=np.float32)

    mean_s = X_source.mean(axis=0)
    mean_t = X_target_calib.mean(axis=0)
    Xs_c   = X_source - mean_s
    Xt_c   = X_target_calib - mean_t

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

    if X_target_final is None:
        return Xs_aligned, Xt_aligned      # سلوك قديم، متوافق للخلف

    # FIX-DA-ITEM4: نفس mean_t (من calib فقط) وPt (PCA مُدرَّب على
    # calib فقط) — الاختبار النهائي لا يُستخدم إطلاقاً في التدريب.
    X_target_final = np.asarray(X_target_final, dtype=np.float32)
    Xt_final_c = X_target_final - mean_t
    Xt_final_aligned = (Xt_final_c @ Pt + mean_t_sub).astype(np.float32)
    return Xs_aligned, Xt_aligned, Xt_final_aligned


# ====================================================================
# 5. قياس انحراف المجال
# ====================================================================
def covariance_frobenius(X_source, X_target, max_samples=5_000,
                         max_features=2_000, random_state=42):
    """
    مقياس تشخيصي فقط (لا يدخل في أي قرار تدريب/تقييم، ولا في CORAL نفسه
    الذي يبقى يعمل على الأبعاد الكاملة — هذا التبسيط هنا فقط لأنه رقم
    وصفي واحد يُبلَّغ عنه).

    FIX (بلاغ MemoryError على 49,978×9,996): إضافة لتصغير العينات
    (max_samples)، الآن أيضاً نُصغِّر عدد الخصائص (أعمدة) عشوائياً إلى
    max_features قبل حساب مصفوفة التغاير F×F — لأن هذه المصفوفة نفسها
    (F×F) هي التكلفة المهيمنة على الذاكرة عند F~10,000 (~400MB float32
    للمصفوفة الواحدة، وهناك اثنتان Cs وCt)، بصرف النظر عن عدد العينات.
    أخذ عيّنة عشوائية من 2,000 خاصية من أصل ~10,000 يُعطي تقديراً تقريبياً
    لكن كافياً تماماً لمقارنة "قبل/بعد" التكيّف كرقم تشخيصي وحيد — وهو
    غير مستخدم في أي مكان آخر من الأنبوب.
    """
    n_s_full = X_source.shape[0] if hasattr(X_source, 'shape') else len(X_source)
    n_t_full = X_target.shape[0] if hasattr(X_target, 'shape') else len(X_target)
    if n_s_full < 3 or n_t_full < 3:
        return np.nan

    rng = np.random.RandomState(random_state)
    # أخذ العينة الفرعية أولاً (على الفهارس)، قبل أي تحويل نوع بيانات
    if n_s_full > max_samples:
        idx_s = rng.choice(n_s_full, max_samples, replace=False)
        X_source = np.asarray(X_source)[idx_s]
    if n_t_full > max_samples:
        idx_t = rng.choice(n_t_full, max_samples, replace=False)
        X_target = np.asarray(X_target)[idx_t]

    F = X_source.shape[1] if hasattr(X_source, 'shape') else len(X_source[0])
    if F > max_features:
        feat_idx = rng.choice(F, max_features, replace=False)
        X_source = np.asarray(X_source)[:, feat_idx]
        X_target = np.asarray(X_target)[:, feat_idx]

    try:
        Xs = np.asarray(X_source, dtype=np.float32)
        Xt = np.asarray(X_target, dtype=np.float32)
        Xs_c = Xs - Xs.mean(axis=0)
        Xt_c = Xt - Xt.mean(axis=0)
        Cs = (Xs_c.T @ Xs_c) / max(Xs_c.shape[0] - 1, 1)
        Ct = (Xt_c.T @ Xt_c) / max(Xt_c.shape[0] - 1, 1)
        return float(np.linalg.norm(Cs - Ct, ord='fro'))
    except Exception as e:
        warnings.warn(f"covariance_frobenius failed: {e}")
        return np.nan


def mmd_squared_linear(X_source, X_target, max_samples=20_000,
                       random_state=42):
    """
    FIX: نفس مشكلة covariance_frobenius (float64 على مصفوفة كاملة قبل
    أي تصغير). هنا الحساب أبسط (فرق المتوسطات فقط) فالحد الآمن أعلى
    (20k)، لكن ما زلنا نتجنب float64 على مصفوفات عريضة (~10k عمود).
    """
    n_s_full = X_source.shape[0] if hasattr(X_source, 'shape') else len(X_source)
    n_t_full = X_target.shape[0] if hasattr(X_target, 'shape') else len(X_target)
    if n_s_full == 0 or n_t_full == 0:
        return np.nan

    rng = np.random.RandomState(random_state)
    if n_s_full > max_samples:
        idx_s = rng.choice(n_s_full, max_samples, replace=False)
        X_source = np.asarray(X_source)[idx_s]
    if n_t_full > max_samples:
        idx_t = rng.choice(n_t_full, max_samples, replace=False)
        X_target = np.asarray(X_target)[idx_t]

    X_source = np.asarray(X_source, dtype=np.float32)
    X_target = np.asarray(X_target, dtype=np.float32)
    delta = X_source.mean(axis=0) - X_target.mean(axis=0)
    return float(delta.astype(np.float64) @ delta.astype(np.float64))


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
