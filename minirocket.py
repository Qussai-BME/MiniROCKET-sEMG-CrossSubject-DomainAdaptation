"""
minirocket.py — Paper 2 V2: GOLDEN MINIROCKET TURBO v6.0
=========================================================
النسخة الذهبية النهائية لخوارزمية MiniROCKET المستخدمة في الورقة.

تعتمد هذه النسخة على الـ v6.0 TURBO من الورقة الأولى مع تحسينات إضافية:

الميزات الأساسية:
  1. استخراج خصائص PPV (Proportion of Positive Values) باستخدام نوى التفاف عشوائية.
  2. استخدام BLAS matmul لتسريع عمليات الالتفاف (أسرع 10 مرات من الحلقات).
  3. معالجة مجمّعة (Batched) لتجنب تجاوز الذاكرة على الإشارات الطويلة.
  4. توليد نوى بنمط وزن محدد (Deterministic weight patterns) لضمان إمكانية التكرار.
  5. حساب biases باستخدام عينة تمثيلية من البيانات (لتوفير الوقت).
  6. دعم متعدد القنوات (Multi-channel) مع تخصيص نوى لكل قناة.
  7. Pipeline متكامل مع StandardScaler و RidgeClassifierCV (مع دعم sklearn و numpy fallback).

التحسينات مقارنة بالنسخ السابقة:
  - تحسين خوارزمية الالتفاف باستخدام مضاعفة المصفوفات بدلاً من الحلقات المتداخلة.
  - إضافة تقرير تقدم مفصل أثناء حساب biases والتحويل (لتتبع وقت التشغيل).
  - معالجة ديناميكية لحجم الدفعة (batch size) في حالة نفاد الذاكرة.
  - استخدام `np.nan_to_num` لضمان عدم وجود قيم شاذة (NaN/Inf) في الخصائص النهائية.

الإصدار: 6.0 TURBO Gold
"""
import numpy as np
import gc
import time
import sys
import traceback
import warnings

# محاولة استيراد sklearn (اختياري) مع التعامل مع عدم وجوده
try:
    from sklearn.linear_model import RidgeClassifierCV as SklearnRidgeCV
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn(
        "scikit-learn not installed. Using numpy-only RidgeClassifierCV fallback. "
        "Install scikit-learn for better performance and stability.",
        ImportWarning
    )

# ======================================================================
# 1.  MiniRocket — محول الخصائص الأساسي (Feature Transformer)
# ======================================================================
class MiniRocket:
    """
    MiniROCKET: تحويل الالتفاف العشوائي للنوى (Random Convolutional Kernel Transform).

    تقوم هذه الفئة باستخراج خصائص PPV (نسبة القيم الموجبة) من الإشارات
    باستخدام آلاف النوى العشوائية. وهي أسرع بكثير من ROCKET الأصلية
    لأنها تستخدم نوى ذات وزن واحد (ثنائي) وتعمل على قناة واحدة لكل نواة.

    المعاملات:
        num_kernels (int): عدد النوى (الافتراضي 10000).
        kernel_length (int): طول النواة (الافتراضي 9).
        proportion (float): نسبة الكمية المستخدمة لحساب bias (الافتراضي 0.5).
        n_channels (int): عدد القنوات في الإشارة (يُحدد تلقائياً أثناء التدريب).
        random_state (int): بذرة العشوائية لإمكانية التكرار.

    الخصائص (Attributes):
        dilations (np.ndarray): قيم التمدد (dilation) لكل نواة.
        weights (np.ndarray): أوزان النوى (قيم +1 أو -1).
        channels (np.ndarray): القناة المخصصة لكل نواة.
        biases (np.ndarray): قيم العتبة (bias) لكل نواة.
        n_channels_ (int): عدد القنوات المستخدمة فعلياً.
    """

    VERSION = "6.0 TURBO Gold"

    def __init__(self, num_kernels=10000, kernel_length=9, proportion=0.5,
                 n_channels=None, random_state=42):
        self.num_kernels = num_kernels
        self.kernel_length = kernel_length
        self.proportion = proportion
        self.n_channels = n_channels
        self.random_state = random_state
        self._fitted = False

        # حدود أداء (لتجنب استهلاك الذاكرة)
        self.BIAS_SAMPLE_LIMIT = 500      # عدد العينات المستخدمة لحساب biases
        self.TRANSFORM_BATCH = 500        # حجم الدفعة الأولي للتحويل

        # سيتم تعيينها أثناء التدريب
        self.dilations = None
        self.weights = None
        self.channels = None
        self.biases = None
        self.unique_dilations = None
        self.n_channels_ = None
        self.input_length_ = None

    # ------------------------------------------------------------------
    # 1.1 توليد النوى (Kernel Generation)
    # ------------------------------------------------------------------
    def _generate_kernels(self, T):
        """
        توليد النوى العشوائية بناءً على طول الإشارة (T).

        تستخدم هذه الدالة خوارزمية MiniROCKET الأصلية لتوزيع قيم التمدد
        بشكل هندسي، ثم توليد أنماط الوزن الثنائية (±1) بشكل حتمي.

        المعاملات:
            T (int): طول الإشارة (عدد العينات).

        المخرجات: (تُخزَّن في خصائص الكائن)
            self.dilations, self.weights, self.channels
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        C = self.n_channels
        L = self.kernel_length

        # حساب قيم التمدد (geometric progression)
        # أقصى تمدد ممكن يضمن أن النواة لا تتجاوز طول الإشارة
        max_power = int(np.floor(np.log2((T - 1) / max(L - 1, 1))))
        num_dilations = max(max_power, 1)
        self.unique_dilations = np.array([2 ** i for i in range(num_dilations)],
                                         dtype=np.int32)

        # توليد أنماط الوزن (±1) بطريقة حتمية
        # هذه الطريقة تضمن إمكانية تكرار النتائج عبر التشغيلات المختلفة
        _num_possible = 2 ** (int(np.ceil(np.log2(L + 1))) - 1)
        all_combos = np.array(
            [[int(w) for w in format(i, f'0{L}b')]
             for i in range(2 ** L)],
            dtype=np.float32
        )
        all_combos = 2.0 * all_combos - 1.0  # تحويل {0,1} إلى {-1,+1}
        weight_patterns = all_combos[:: 2**L // _num_possible]

        # توزيع النوى على قيم التمدد بالتساوي
        base_k = self.num_kernels // num_dilations
        remainder = self.num_kernels % num_dilations

        dilations_list = []
        weights_list = []
        channels_list = []
        kernel_idx = 0

        for d_idx, d in enumerate(self.unique_dilations):
            n_k = base_k + (1 if d_idx < remainder else 0)
            for _ in range(n_k):
                dilations_list.append(d)
                # اختيار قناة عشوائية لهذه النواة
                channels_list.append(np.random.randint(0, C))
                # اختيار نمط وزن من الأنماط المتاحة (دوري)
                w = weight_patterns[kernel_idx % _num_possible].copy()
                weights_list.append(w)
                kernel_idx += 1

        # تحويل القوائم إلى مصفوفات numpy
        self.dilations = np.array(dilations_list, dtype=np.int32)
        self.weights = np.array(weights_list, dtype=np.float32)
        self.channels = np.array(channels_list, dtype=np.int32)
        self.biases = np.zeros(self.num_kernels, dtype=np.float32)
        self.n_channels_ = C
        self.input_length_ = T

        print(f"    [MiniRocket] Generated {self.num_kernels} kernels "
              f"({num_dilations} dilations, {_num_possible} weight patterns, "
              f"d=[{','.join(str(int(x)) for x in self.unique_dilations)}])",
              flush=True)

    # ------------------------------------------------------------------
    # 1.2 حساب Biases (Quantiles)
    # ------------------------------------------------------------------
    def _compute_biases(self, X):
        """
        حساب قيمة bias لكل نواة باستخدام عينة تمثيلية من البيانات.

        تستخدم هذه الدالة عينة فرعية (حتى BIAS_SAMPLE_LIMIT عينة) لحساب
        الكميات (quantiles) لنتائج الالتفاف، مما يوفر وقتاً كبيراً مع الحفاظ
        على دقة مقبولة.

        المعاملات:
            X (np.ndarray): مصفوفة الإشارات (n_samples, n_channels, T).
        """
        N, C, T = X.shape
        L = self.kernel_length

        # اختيار عينة تمثيلية (لتسريع الحساب)
        N_sub = min(N, self.BIAS_SAMPLE_LIMIT)
        if N_sub < N:
            rng = np.random.RandomState(self.random_state + 1 if self.random_state else 0)
            idx = rng.choice(N, N_sub, replace=False)
        else:
            idx = np.arange(N)
        X_sub = np.ascontiguousarray(X[idx], dtype=np.float32)

        t0 = time.time()
        n_groups = len(self.unique_dilations)

        print(f"    [MiniRocket] Computing biases on {N_sub} samples...", flush=True)

        for g_idx, d in enumerate(self.unique_dilations):
            dt0 = time.time()
            d_int = int(d)

            mask = (self.dilations == d)
            global_idx = np.where(mask)[0]
            K_d = len(global_idx)
            T_out = T - (L - 1) * d_int

            if T_out <= 0:
                continue

            W_d = self.weights[mask]
            ch_d = self.channels[mask]

            # معالجة كل قناة على حدة لتوفير الذاكرة
            for c in range(C):
                c_mask = (ch_d == c)
                K_c = int(c_mask.sum())
                if K_c == 0:
                    continue

                W_c = W_d[c_mask]
                gidx_c = global_idx[c_mask]
                X_c = X_sub[:, c, :]

                # بناء مصفوفة التأخيرات (lag matrix) باستخدام as_strided
                # هذه الطريقة أسرع من الحلقات
                lag_flat = np.empty((L, N_sub * T_out), dtype=np.float32)
                for j in range(L):
                    lag_flat[j] = X_c[:, j * d_int : j * d_int + T_out].ravel()

                # الالتفاف عبر مضاعفة المصفوفات (BLAS)
                conv = W_c @ lag_flat  # (K_c, N_sub * T_out)

                # حساب الكميات (quantiles) لكل نواة
                quantiles = np.quantile(conv, self.proportion, axis=1)
                self.biases[gidx_c] = -quantiles.astype(np.float32)

                # تنظيف الذاكرة المؤقتة
                del lag_flat, conv

            dt = time.time() - dt0
            elapsed = time.time() - t0
            eta = (elapsed / (g_idx + 1)) * (n_groups - g_idx - 1) if g_idx > 0 else dt * (n_groups - 1)
            print(f"    [Bias] {g_idx+1}/{n_groups}: d={d}, {K_d} kernels, "
                  f"{dt:.1f}s (ETA {eta:.0f}s)", flush=True)
            gc.collect()

        print(f"    [MiniRocket] All biases computed ({time.time()-t0:.1f}s)", flush=True)

    # ------------------------------------------------------------------
    # 1.3 التحويل المجمّع (Batched Transform)
    # ------------------------------------------------------------------
    def _batched_transform(self, X):
        """
        تحويل الإشارات إلى خصائص PPV باستخدام المعالجة المجمّعة.

        تقوم هذه الدالة بتطبيق جميع النوى على الإشارات، وحساب نسبة القيم
        الموجبة (PPV) لكل نواة، مع تقسيم العمل إلى دفعات لتجنب تجاوز الذاكرة.

        المعاملات:
            X (np.ndarray): مصفوفة الإشارات (n_samples, n_channels, T).

        المخرجات:
            np.ndarray: خصائص PPV (n_samples, num_kernels * n_channels).
        """
        N, C, T = X.shape
        X = np.ascontiguousarray(X, dtype=np.float32)
        K = self.num_kernels
        L = self.kernel_length

        # مصفوفة النتائج النهائية (PPV)
        ppv = np.zeros((N, K), dtype=np.float32)

        t0 = time.time()
        n_groups = len(self.unique_dilations)
        batch_size = self.TRANSFORM_BATCH

        print(f"    [MiniRocket] Transforming {N} samples, {K} kernels...", flush=True)

        for g_idx, d in enumerate(self.unique_dilations):
            dt0 = time.time()
            d_int = int(d)

            mask = (self.dilations == d)
            global_idx = np.where(mask)[0]
            K_d = len(global_idx)
            T_out = T - (L - 1) * d_int

            if T_out <= 0:
                continue

            W_d = self.weights[mask]
            b_d = self.biases[mask]
            ch_d = self.channels[mask]

            for c in range(C):
                c_mask = (ch_d == c)
                K_c = int(c_mask.sum())
                if K_c == 0:
                    continue

                W_c = W_d[c_mask]
                b_c = b_d[c_mask]
                gidx_c = global_idx[c_mask]
                X_c = X[:, c, :]

                # المعالجة المجمّعة (Batch processing)
                sb = 0
                while sb < N:
                    eb = min(sb + batch_size, N)
                    B = eb - sb
                    X_sub = X_c[sb:eb]

                    try:
                        # بناء مصفوفة التأخيرات للدفعة الحالية
                        lag_flat = np.empty((L, B * T_out), dtype=np.float32)
                        for j in range(L):
                            lag_flat[j] = X_sub[:, j * d_int : j * d_int + T_out].ravel()

                        # الالتفاف عبر مضاعفة المصفوفات (BLAS)
                        conv = W_c @ lag_flat  # (K_c, B * T_out)
                        # إضافة bias (البث التلقائي)
                        conv += b_c[:, None]

                        # حساب PPV: نسبة القيم > 0 عبر المحور الزمني
                        ppv_sub = np.mean(
                            (conv > 0).reshape(K_c, B, T_out), axis=2
                        )  # (K_c, B)

                        # تخزين النتائج في المصفوفة النهائية
                        ppv[sb:eb, gidx_c] = ppv_sub.T  # (B, K_c)

                        # تنظيف الذاكرة المؤقتة
                        del lag_flat, conv

                    except MemoryError:
                        # في حال نفاد الذاكرة، نقوم بتقليل حجم الدفعة
                        if batch_size <= 32:
                            print(f"\n    [ERROR] MemoryError at batch_size={batch_size}")
                            traceback.print_exc()
                            break
                        old_bs = batch_size
                        batch_size = max(32, batch_size // 2)
                        print(f"\n    [Memory] Reducing batch size from {old_bs} to {batch_size}")
                        gc.collect()
                        # إعادة المحاولة باستخدام الدفعة الجديدة (نستمر من نفس الموضع)
                        continue

                    sb = eb

            # تقرير التقدم
            dt = time.time() - dt0
            elapsed = time.time() - t0
            eta = (elapsed / (g_idx + 1)) * (n_groups - g_idx - 1) if g_idx > 0 else dt * (n_groups - 1)
            print(f"    [Transform] {g_idx+1}/{n_groups}: d={d}, {K_d} kernels "
                  f"({dt:.1f}s, total ETA {eta:.0f}s)", flush=True)
            gc.collect()

        # معالجة القيم الشاذة (NaN, Inf)
        ppv = np.nan_to_num(ppv, nan=0.0, posinf=1.0, neginf=0.0)
        return ppv

    # ------------------------------------------------------------------
    # 1.4 واجهات التدريب والتحويل (Fit / Transform)
    # ------------------------------------------------------------------
    def fit(self, X, y=None):
        """
        تدريب المحول: توليد النوى وحساب biases.

        المعاملات:
            X (np.ndarray): الإشارات (n_samples, n_channels, T).
            y (ignored): غير مستخدم (للتوافق مع sklearn).

        المخرجات:
            self (MiniRocket): الكائن نفسه بعد التدريب.
        """
        X = np.asarray(X, dtype=np.float32)
        N, C, T = X.shape

        # تحديد عدد القنوات تلقائياً
        if self.n_channels is None:
            self.n_channels = C
        elif self.n_channels != C:
            warnings.warn(
                f"n_channels mismatch: expected {self.n_channels}, got {C}. "
                f"Using {C}."
            )
            self.n_channels = C

        # توليد النوى
        self._generate_kernels(T)

        # حساب biases
        self._compute_biases(X)

        self._fitted = True
        return self

    def transform(self, X):
        """
        تحويل الإشارات إلى خصائص PPV باستخدام النوى المدربة.

        المعاملات:
            X (np.ndarray): الإشارات (n_samples, n_channels, T).

        المخرجات:
            np.ndarray: خصائص PPV (n_samples, num_kernels * n_channels).
        """
        if not self._fitted:
            raise RuntimeError(
                "MiniRocket must be fitted before transform. Call fit() or fit_transform() first."
            )
        X = np.asarray(X, dtype=np.float32)
        return self._batched_transform(X)

    def fit_transform(self, X, y=None):
        """
        تدريب المحول وتحويل الإشارات في خطوة واحدة.

        المعاملات:
            X (np.ndarray): الإشارات (n_samples, n_channels, T).
            y (ignored): غير مستخدم.

        المخرجات:
            np.ndarray: خصائص PPV (n_samples, num_kernels * n_channels).
        """
        self.fit(X, y)
        return self.transform(X)


# ======================================================================
# 2.  RidgeClassifierCV — المصنف مع اختيار تلقائي لـ alpha
# ======================================================================
class RidgeClassifierCV:
    """
    مصنف Ridge مع اختيار تلقائي لمعامل التنظيم (alpha) باستخدام GCV.

    هذه الفئة توفّر واجهة موحدة بغض النظر عن وجود sklearn أم لا.
    في حال توفر sklearn، تستخدم `RidgeClassifierCV` الأصلي (الأفضل).
    في حال عدم التوفر، تستخدم تطبيقاً بـ numpy يعتمد على تحليل القيم الذاتية.

    المعاملات:
        alphas (list): قيم alpha المراد اختبارها.
        random_state (int): بذرة العشوائية (تُستخدم في sklearn).
    """

    VERSION = "3.0 Gold"

    def __init__(self, alphas=None, random_state=42):
        if HAS_SKLEARN:
            self.alphas = alphas or np.logspace(-3, 6, 25).tolist()
            self._sklearn = SklearnRidgeCV(
                alphas=self.alphas,
                scoring='accuracy',
                cv=None,  # GCV (Leave-One-Out)
            )
            self._backend = 'sklearn'
        else:
            self.alphas = alphas or np.logspace(-3, 6, 20).tolist()
            self._sklearn = None
            self._backend = 'numpy'

        self.random_state = random_state
        self.classes_ = None
        self.coef_ = None
        self.intercept_ = None
        self.alpha_ = None

    def fit(self, X, y):
        """
        تدريب المصنف.

        المعاملات:
            X (np.ndarray): خصائص التدريب (n_samples, n_features).
            y (np.ndarray): التسميات (n_samples,).

        المخرجات:
            self (RidgeClassifierCV): الكائن نفسه بعد التدريب.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        if self._backend == 'sklearn':
            self._sklearn.fit(X, y)
            self.classes_ = self._sklearn.classes_
            self.coef_ = self._sklearn.coef_
            self.intercept_ = self._sklearn.intercept_
            self.alpha_ = self._sklearn.alpha_
            train_acc = self._sklearn.score(X, y)
            print(f"    [RidgeClassifierCV] SKLEARN alpha={self.alpha_:.4f}, "
                  f"{len(self.classes_)} classes, train_acc={train_acc:.4f}", flush=True)
        else:
            self._fit_numpy(X, y)

        return self

    def _fit_numpy(self, X, y):
        """
        تطبيق numpy الاحتياطي لـ RidgeClassifierCV.

        يستخدم تحليل القيم الذاتية (Eigenvalue Decomposition) لحساب GCV
        واختيار أفضل alpha.
        """
        N, P = X.shape
        self.classes_ = np.unique(y)
        K = len(self.classes_)

        # تحويل التسميات إلى مصفوفة ثنائية (One-hot encoding)
        Y = np.zeros((N, K), dtype=np.float64)
        for i, c in enumerate(self.classes_):
            Y[y == c, i] = 1.0

        # توسيط البيانات
        X_mean = X.mean(axis=0)
        Y_mean = Y.mean(axis=0)
        Xc = X - X_mean
        Yc = Y - Y_mean

        # حساب المصفوفات
        XTX = Xc.T @ Xc
        XTY = Xc.T @ Yc

        best_alpha = self.alphas[0]
        best_score = np.inf

        # إذا كان عدد الخصائص صغيراً، نستخدم تحليل القيم الذاتية السريع
        if P <= 5000:
            eigvals, V = np.linalg.eigh(XTX)
            eigvals = np.maximum(eigvals, 0)  # تجنب القيم السالبة بسبب الخطأ العددي
            VTY = V.T @ XTY

            for alpha in self.alphas:
                D_inv = 1.0 / (eigvals + alpha)
                coef = V @ (D_inv[:, None] * VTY)
                residuals = Yc - Xc @ coef
                RSS = np.sum(residuals ** 2)
                df = np.sum(eigvals / (eigvals + alpha))
                gcv = RSS * N / max(N - df, 1) ** 2
                if np.isfinite(gcv) and gcv < best_score:
                    best_score = gcv
                    best_alpha = alpha

            del eigvals, V, VTY

        else:
            # للخصائص الكثيرة، نستخدم حل مباشر (أبطأ لكن أقل استهلاكاً للذاكرة)
            eye_P = np.eye(P, dtype=np.float64)
            for alpha in self.alphas:
                try:
                    coef = np.linalg.solve(XTX + alpha * eye_P, XTY)
                    RSS = float(np.sum((Yc - Xc @ coef) ** 2))
                    if RSS < best_score:
                        best_score = RSS
                        best_alpha = alpha
                except np.linalg.LinAlgError:
                    continue

        # الحل النهائي بأفضل alpha
        eye_P = np.eye(P, dtype=np.float64)
        self.coef_ = np.linalg.solve(XTX + best_alpha * eye_P, XTY)
        self.intercept_ = Y_mean - X_mean @ self.coef_
        self.alpha_ = best_alpha

        # حساب دقة التدريب
        pred = np.argmax(Xc @ self.coef_, axis=1)
        true = np.argmax(Yc, axis=1)
        train_acc = float(np.mean(pred == true))
        print(f"    [RidgeClassifierCV] NUMPY alpha={best_alpha:.4f}, "
              f"{K} classes, train_acc={train_acc:.4f}", flush=True)

        # تنظيف الذاكرة
        del XTX, XTY, Xc, Yc
        gc.collect()

    def decision_function(self, X):
        """
        حساب درجات القرار (scores) لكل صنف.

        المعاملات:
            X (np.ndarray): خصائص الاختبار (n_samples, n_features).

        المخرجات:
            np.ndarray: درجات القرار (n_samples, n_classes).
        """
        X = np.asarray(X, dtype=np.float64)
        if self._backend == 'sklearn':
            return self._sklearn.decision_function(X)
        return X @ self.coef_ + self.intercept_

    def predict(self, X):
        """
        توقع التسميات.

        المعاملات:
            X (np.ndarray): خصائص الاختبار (n_samples, n_features).

        المخرجات:
            np.ndarray: التسميات المتوقعة (n_samples,).
        """
        if self._backend == 'sklearn':
            return self._sklearn.predict(X)
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def score(self, X, y):
        """
        حساب دقة التصنيف.

        المعاملات:
            X (np.ndarray): خصائص الاختبار (n_samples, n_features).
            y (np.ndarray): التسميات الحقيقية (n_samples,).

        المخرجات:
            float: دقة التصنيف.
        """
        y = np.asarray(y)
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))


# ======================================================================
# 3.  MiniRocketPipeline — الأنبوب المتكامل (MiniRocket + Scaler + Ridge)
# ======================================================================
class MiniRocketPipeline:
    """
    أنبوب متكامل يضم:
        1. MiniRocket (استخراج خصائص PPV)
        2. StandardScaler (تقييس الخصائص)
        3. RidgeClassifierCV (تصنيف مع اختيار تلقائي لـ alpha)

    هذا الأنبوب يبسط عملية التدريب والاختبار بشكل كبير، ويضمن
    تطبيق نفس التحويلات على بيانات التدريب والاختبار.

    المعاملات:
        num_kernels (int): عدد النوى في MiniRocket.
        kernel_length (int): طول النواة.
        proportion (float): نسبة الكمية لحساب bias.
        alphas (list): قيم alpha لـ RidgeClassifierCV.
        random_state (int): بذرة العشوائية.
    """

    VERSION = "7.0 GOLD Pipeline"

    def __init__(self, num_kernels=10000, kernel_length=9, proportion=0.5,
                 alphas=None, random_state=42):
        # مكونات الأنبوب
        self.rocket = MiniRocket(
            num_kernels=num_kernels,
            kernel_length=kernel_length,
            proportion=proportion,
            random_state=random_state,
        )

        # StandardScaler (إن وجد)
        if HAS_SKLEARN:
            self.scaler = StandardScaler()
            print("    [Pipeline] Using sklearn StandardScaler + RidgeClassifierCV", flush=True)
        else:
            self.scaler = None
            print("    [Pipeline] Using numpy-only RidgeClassifierCV (no sklearn)", flush=True)

        self.clf = RidgeClassifierCV(alphas=alphas, random_state=random_state)

    def fit(self, X, y):
        """
        تدريب الأنبوب بالكامل.

        المعاملات:
            X (np.ndarray): الإشارات (n_samples, n_channels, T).
            y (np.ndarray): التسميات (n_samples,).

        المخرجات:
            self (MiniRocketPipeline): الكائن نفسه بعد التدريب.
        """
        print(f"    [Pipeline] Fitting on {X.shape[0]} samples, {X.shape[1]} channels, "
              f"{X.shape[2]} timesteps", flush=True)

        # 1. استخراج خصائص PPV
        X_feat = self.rocket.fit_transform(X, y)

        # تشخيص الخصائص الخام
        print(f"    [Diag] Raw PPV: mean={X_feat.mean():.4f}, std={X_feat.std():.4f}, "
              f"min={X_feat.min():.4f}, max={X_feat.max():.4f}", flush=True)

        # 2. تقييس الخصائص (StandardScaler)
        if self.scaler is not None:
            X_feat = self.scaler.fit_transform(X_feat)
            print(f"    [Diag] Scaled PPV: mean={X_feat.mean():.4f}, std={X_feat.std():.4f}",
                  flush=True)

        # 3. تدريب المصنف Ridge
        self.clf.fit(X_feat, y)

        return self

    def predict(self, X):
        """
        توقع التسميات لبيانات جديدة.

        المعاملات:
            X (np.ndarray): الإشارات (n_samples, n_channels, T).

        المخرجات:
            np.ndarray: التسميات المتوقعة (n_samples,).
        """
        # 1. استخراج الخصائص
        X_feat = self.rocket.transform(X)

        # 2. تقييس الخصائص
        if self.scaler is not None:
            X_feat = self.scaler.transform(X_feat)

        # 3. التصنيف
        return self.clf.predict(X_feat)

    def score(self, X, y):
        """
        حساب دقة التصنيف.

        المعاملات:
            X (np.ndarray): الإشارات (n_samples, n_channels, T).
            y (np.ndarray): التسميات الحقيقية (n_samples,).

        المخرجات:
            float: دقة التصنيف.
        """
        y = np.asarray(y)
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))

    def decision_function(self, X):
        """
        حساب درجات القرار.

        المعاملات:
            X (np.ndarray): الإشارات (n_samples, n_channels, T).

        المخرجات:
            np.ndarray: درجات القرار (n_samples, n_classes).
        """
        X_feat = self.rocket.transform(X)
        if self.scaler is not None:
            X_feat = self.scaler.transform(X_feat)
        return self.clf.decision_function(X_feat)

    @property
    def classes_(self):
        """إرجاع الأصناف المستخدمة في التدريب."""
        return self.clf.classes_

    @property
    def n_channels_(self):
        """إرجاع عدد القنوات المستخدمة بواسطة MiniRocket."""
        return self.rocket.n_channels_

    @property
    def dilations_(self):
        """إرجاع قيم التمدد لكل نواة."""
        return self.rocket.dilations

    @property
    def coef_(self):
        """إرجاع معاملات المصنف (للتحليل والتفسير)."""
        return self.clf.coef_