"""
feature_extractors.py — 
========================================================================
يوفر هذا الملف نقطة وصول واحدة لاختيار مستخرج الخصائص:

  "canonical"   -> MiniRocketMultivariate القانوني (حزمة sktime، وهي
                   الامتداد الرسمي لتطبيق المؤلفين الأصليين، Dempster et
                   al.). هذا هو المستخرج **الأساسي** بعد التدقيق.

  "custom_ppv"  -> minirocket.MiniRocket، التطبيق اليدوي المخصص المستخدم
                   في مسودة الورقة الأولى. يُبقى عليه بوضوح كـ **مقارن
                   موسوم** (labeled comparator) — لا يُحذف، وفق طلب
                   المشرف الحرفي: "retaining the present PPV
                   implementation as a clearly labeled comparator".

كلا المستخرجين يعرضان نفس الواجهة (fit / transform / fit_transform) على
نفس شكل الدخل (n_samples, n_channels, n_timepoints)، بحيث يمكن لبقية
الـ pipeline (p01a إلخ) استبدال أحدهما بالآخر دون أي تغيير آخر.

التثبيت (على جهاز Qussai، مرة واحدة):
    pip install sktime numba

ملاحظة أداء: أول استدعاء لـ fit_transform على كل عملية Python بطيء بسبب
JIT compilation في numba (~20-30s). هذا اقتصادي طالما العملية تعالج عدة
LOSO folds بدل fold واحد فقط لكل عملية.
"""
import warnings
import numpy as np

from minirocket import MiniRocket as _CustomPPVMiniRocket

try:
    from sktime.transformations.panel.rocket import MiniRocketMultivariate
    HAS_SKTIME = True
except ImportError:
    HAS_SKTIME = False


class CanonicalMiniRocket:
    """
    غلاف (wrapper) حول sktime.MiniRocketMultivariate يعرض نفس واجهة
    minirocket.MiniRocket (fit / transform / fit_transform) على مصفوفات
    numpy بشكل (n_samples, n_channels, n_timepoints) — نفس الشكل المستخدم
    في كل هذا المشروع، دون أي تحويل إضافي مطلوب من المستدعي.

    FIX (بعد بلاغ خطأ MemoryError على 50k عينة × 10k kernel، Windows):
    sktime's MiniRocketMultivariate._transform يبني مصفوفة وسيطة كبيرة
    لكل الدفعة (batch) دفعة واحدة — على عكس minirocket.py المخصص الذي
    يُقسّم داخلياً (_batched_transform). لذلك transform() هنا يُقسّم
    العينات يدوياً إلى دفعات صغيرة (batch_size، افتراضياً 1000) قبل
    تمريرها لـ sktime، ثم يُلصق (vstack) النتائج — هذا لا يُغيّر القيم
    الناتجة إطلاقاً (transform لكل عينة مستقل عن العينات الأخرى بعد
    fit)، فقط يُقيّد الذاكرة المُستخدمة في أي لحظة.
    """

    def __init__(self, num_kernels=10000, random_state=42, n_jobs=1,
                batch_size=1000, fit_mode="full_train",
                max_fit_samples=2000, **_ignored):
        if not HAS_SKTIME:
            raise ImportError(
                "CanonicalMiniRocket requires 'sktime' (and 'numba'). "
                "Install with: pip install sktime numba"
            )
        self.num_kernels = num_kernels
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        if fit_mode not in ("full_train", "sampled_train"):
            raise ValueError("fit_mode must be 'full_train' or 'sampled_train'")
        self.fit_mode = fit_mode
        self.max_fit_samples = int(max_fit_samples)
        self._transformer = MiniRocketMultivariate(
            num_kernels=num_kernels,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self._fitted = False

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        # full_train هو الوضع العلمي الافتراضي. sampled_train متاح فقط
        # للاختبارات/benchmarks؛ لا يجوز تفعيله ضمنيًا لتوفير الذاكرة.
        if self.fit_mode == "sampled_train" and X.shape[0] > self.max_fit_samples:
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(X.shape[0], self.max_fit_samples, replace=False)
            self._transformer.fit(X[idx])
        else:
            self._transformer.fit(X)
        self._fitted = True
        return self

    def transform(self, X):
        if not self._fitted:
            raise RuntimeError(
                "CanonicalMiniRocket must be fitted before transform()."
            )
        X = np.asarray(X, dtype=np.float32)
        n = X.shape[0]
        if n <= self.batch_size:
            return np.asarray(self._transformer.transform(X),
                              dtype=np.float32)

        chunks = []
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            chunk_out = np.asarray(
                self._transformer.transform(X[start:end]), dtype=np.float32)
            chunks.append(chunk_out)
        return np.vstack(chunks)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


EXTRACTOR_REGISTRY = {
    "canonical":  CanonicalMiniRocket,
    "custom_ppv": _CustomPPVMiniRocket,
}


def get_feature_extractor(name, num_kernels=10000, random_state=42, **kwargs):
    """
    Factory موحّدة. name يجب أن تكون إحدى config.FEATURE_EXTRACTOR_OPTIONS.

    مثال:
        rocket = get_feature_extractor("canonical", num_kernels=10000,
                                       random_state=42)
        X_feat = rocket.fit_transform(X)   # X: (n, n_channels, n_timepoints)
    """
    if name not in EXTRACTOR_REGISTRY:
        raise ValueError(
            f"Unknown feature_extractor '{name}'. "
            f"Choose from {list(EXTRACTOR_REGISTRY)}."
        )
    if name == "canonical" and not HAS_SKTIME:
        warnings.warn(
            "sktime not installed -> cannot use the canonical extractor "
            "required for the canonical extractor. Falling back to 'custom_ppv' for "
            "this call only. Install sktime+numba before the real run: "
            "pip install sktime numba"
        )
        name = "custom_ppv"
    cls = EXTRACTOR_REGISTRY[name]
    if name == "custom_ppv":
        # The hand-written comparator has its own historical constructor and
        # must not silently inherit canonical-only fit/parallelism arguments.
        return cls(num_kernels=num_kernels, random_state=random_state)
    return cls(num_kernels=num_kernels, random_state=random_state, **kwargs)
