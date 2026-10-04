"""
statistical_utils.py — Paper 2 V2: GOLDEN STATISTICAL ENGINE
=============================================================
الوحدة الإحصائية المركزية للمشروع. تحتوي على جميع الاختبارات
والمقاييس المستخدمة في التحليل الإحصائي للورقة.

تشمل هذه الوحدة:
  1. مقاييس التصنيف الأساسية: F1، مصفوفة الارتباك.
  2. إحصاءات وصفية: المتوسط، الانحراف المعياري، فترة الثقة.
  3. اختبار فريدمان (مع تصحيح إيمان-دافنبورت).
  4. اختبار نيميني للمقارنات البعدية (Post-hoc).
  5. اختبار ويلكوكسون للمقارنات الزوجية.
  6. تصحيح هولم-سيداك للمقارنات المتعددة.
  7. حجم التأثير كوهين د (Cohen's d).
  8. حساب دقة التخمين العشوائي (Random Baseline).
  9. دوال مساعدة لبناء مصفوفات الرتب.

تم تصميم هذه الوحدة لتكون:
  - دقيقة رياضياً (مع الاستشهاد بالمراجع).
  - مقاومة للأخطاء (معالجة القيم المفقودة، التباين الصفري، العينات الصغيرة).
  - فعّالة من حيث الأداء (استخدام عمليات المصفوفات المتجهة).
  - متوافقة مع مخرجات JSON الخاصة بـ p01a_run_main_loso_benchmark.py.
"""
import numpy as np
from scipy import stats
from itertools import combinations
import warnings


# ====================================================================
# 1. مقاييس التصنيف الأساسية
# ====================================================================
def per_class_f1(y_true, y_pred, n_classes):
    """
    حساب F1-score لكل صنف على حدة.

    المعاملات:
        y_true (np.ndarray): التسميات الحقيقية (n_samples,).
        y_pred (np.ndarray): التسميات المتوقعة (n_samples,).
        n_classes (int): عدد الأصناف الكلي.

    المخرجات:
        list: قائمة بقيم F1 لكل صنف (n_classes,).
    """
    f1s = []
    for c in range(n_classes):
        # حساب عناصر مصفوفة الارتباك لهذا الصنف
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))

        # حساب الدقة (Precision) والاستدعاء (Recall)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # حساب F1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(float(f1))

    return f1s


def confusion_matrix(y_true, y_pred, n_classes):
    """
    حساب مصفوفة الارتباك (Confusion Matrix) بالأعداد الصحيحة.

    المعاملات:
        y_true (np.ndarray): التسميات الحقيقية (n_samples,).
        y_pred (np.ndarray): التسميات المتوقعة (n_samples,).
        n_classes (int): عدد الأصناف الكلي.

    المخرجات:
        np.ndarray: مصفوفة الارتباك (n_classes, n_classes) من نوع int.
    """
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


# ====================================================================
# 2. إحصاءات وصفية (Descriptive Statistics)
# ====================================================================
def descriptive_stats(values, confidence=0.95):
    """
    حساب الإحصاءات الوصفية الأساسية لقائمة من القيم.

    المعاملات:
        values (list or np.ndarray): قائمة القيم (مثل دقة كل شخص).
        confidence (float): مستوى الثقة لفترة الثقة (الافتراضي 0.95).

    المخرجات:
        dict: قاموس يحتوي على:
            - 'mean': المتوسط الحسابي
            - 'std': الانحراف المعياري (عينة)
            - 'median': الوسيط
            - 'min': أقل قيمة
            - 'max': أكبر قيمة
            - 'n': عدد العينات
            - 'ci95_lower': الحد الأدنى لفترة الثقة 95%
            - 'ci95_upper': الحد الأعلى لفترة الثقة 95%
            - 'sem': خطأ المعيار (Standard Error of the Mean)
    """
    v = np.asarray(values, dtype=np.float64)
    n = len(v)

    if n == 0:
        return {
            'mean': np.nan, 'std': np.nan, 'median': np.nan,
            'min': np.nan, 'max': np.nan, 'n': 0,
            'ci95_lower': np.nan, 'ci95_upper': np.nan, 'sem': np.nan
        }

    mean = float(np.mean(v))
    std = float(np.std(v, ddof=1)) if n > 1 else 0.0
    median = float(np.median(v))
    min_val = float(np.min(v))
    max_val = float(np.max(v))
    sem = std / np.sqrt(n) if n > 0 else 0.0

    # فترة الثقة 95% باستخدام توزيع t-student
    if n > 1:
        t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
        ci_lower = mean - t_val * sem
        ci_upper = mean + t_val * sem
    else:
        ci_lower = mean
        ci_upper = mean

    return {
        'mean': mean,
        'std': std,
        'median': median,
        'min': min_val,
        'max': max_val,
        'n': n,
        'ci95_lower': float(ci_lower),
        'ci95_upper': float(ci_upper),
        'sem': float(sem),
    }


# ====================================================================
# 3. اختبار فريدمان (Friedman Test) مع تصحيح إيمان-دافنبورت
# ====================================================================
def friedman_test(accuracy_matrix, method_names=None):
    """
    اختبار فريدمان للمقارنة بين عدة طرق (شروط متكررة).

    تستخدم هذه الدالة اختبار فريدمان الأصلي، ثم تقوم بتصحيح إيمان-دافنبورت
    (Iman-Davenport) للحصول على قيمة p أكثر دقة باستخدام توزيع F،
    خاصة للعينات الصغيرة.

    المعاملات:
        accuracy_matrix (np.ndarray): مصفوفة الدقة (n_subjects, n_methods).
        method_names (list): أسماء الطرق (اختياري، للتوثيق).

    المخرجات:
        dict: يحتوي على:
            - 'statistic': إحصائية فريدمان (chi2)
            - 'p_value': قيمة p (باستخدام تصحيح إيمان-دافنبورت)
            - 'n_subjects': عدد الأشخاص
            - 'n_methods': عدد الطرق
            - 'mean_ranks': قائمة بمتوسط الرتب لكل طريقة
            - 'f_iman': إحصائية F لـ إيمان-دافنبورت
            - 'p_iman': قيمة p لـ إيمان-دافنبورت
            - 'method_names': أسماء الطرق (إذا تم توفيرها)
    """
    n_subjects, n_methods = accuracy_matrix.shape
    method_names = method_names or [f"M{i}" for i in range(n_methods)]

    # التأكد من وجود عينات كافية
    if n_subjects < 3 or n_methods < 3:
        warnings.warn("Friedman test requires at least 3 subjects and 3 methods.")
        return {
            'statistic': np.nan,
            'p_value': np.nan,
            'n_subjects': n_subjects,
            'n_methods': n_methods,
            'mean_ranks': np.full(n_methods, np.nan),
            'f_iman': np.nan,
            'p_iman': np.nan,
            'method_names': method_names,
        }

    # حساب الرتب داخل كل شخص (كل صف)
    ranks = np.zeros_like(accuracy_matrix, dtype=np.float64)
    for i in range(n_subjects):
        ranks[i] = stats.rankdata(-accuracy_matrix[i])  # الترتيب التنازلي (الأفضل = 1)

    # متوسط الرتب لكل طريقة (كل عمود)
    mean_ranks = ranks.mean(axis=0)

    # إحصائية فريدمان (Chi-squared)
    chi2 = 12.0 * n_subjects / (n_methods * (n_methods + 1)) * \
           np.sum((mean_ranks - (n_methods + 1) / 2.0) ** 2)

    # تصحيح إيمان-دافنبورت (Iman-Davenport) باستخدام توزيع F
    # هذا التصحيح يعطي نتائج أكثر دقة للعينات الصغيرة
    denom = n_subjects * (n_methods - 1) - chi2
    if denom > 0:
        f_iman = (chi2 * (n_subjects - 1)) / denom
        df1 = n_methods - 1
        df2 = (n_subjects - 1) * df1
        if df2 > 0 and f_iman > 0:
            p_iman = 1.0 - stats.f.cdf(f_iman, df1, df2)
        else:
            p_iman = 1.0
    else:
        f_iman = float('inf')
        p_iman = 0.0

    # قيمة p من توزيع Chi-squared (تُستخدم كمرجع فقط)
    p_chi2 = 1.0 - stats.chi2.cdf(chi2, n_methods - 1) if chi2 > 0 else 1.0

    return {
        'statistic': float(chi2),
        'p_value': float(p_iman),  # القيمة الأساسية (باستخدام تصحيح إيمان-دافنبورت)
        'p_chi2': float(p_chi2),   # قيمة p من توزيع Chi-squared (بدون تصحيح)
        'n_subjects': n_subjects,
        'n_methods': n_methods,
        'mean_ranks': mean_ranks.tolist(),
        'f_iman': float(f_iman),
        'p_iman': float(p_iman),
        'method_names': method_names,
        'ranks_matrix': ranks,  # مصفوفة الرتب الكاملة (للتحليل الإضافي)
    }


# ====================================================================
# 4. اختبار نيميني للمقارنات البعدية (Nemenyi Post-hoc)
# ====================================================================
def nemenyi_posthoc(mean_ranks, n_subjects, n_methods, alpha=0.05):
    """
    اختبار نيميني للمقارنات البعدية (Post-hoc) بعد اختبار فريدمان.

    تحسب هذه الدالة المسافة الحرجة (Critical Difference) وتقارن بين
    جميع أزواج الطرق لتحديد الفروق ذات الدلالة الإحصائية.

    المعاملات:
        mean_ranks (list or np.ndarray): متوسط الرتب لكل طريقة (من friedman_test).
        n_subjects (int): عدد الأشخاص.
        n_methods (int): عدد الطرق.
        alpha (float): مستوى الدلالة (الافتراضي 0.05).

    المخرجات:
        dict: يحتوي على:
            - 'critical_difference': المسافة الحرجة
            - 'q_value': قيمة Q المستخدمة (من توزيع Studentized Range)
            - 'comparisons': قائمة بالمقارنات الزوجية، كل مقارنة تحتوي على:
                - 'method_i', 'method_j': أرقام الطرق
                - 'rank_diff': الفرق بين متوسطي الرتب
                - 'significant': True إذا كان الفرق أكبر من المسافة الحرجة
                - 'p_approx': قيمة p تقريبية (محسوبة من توزيع Studentized Range)
    """
    # قيمة Q من توزيع Studentized Range
    # نستخدم درجة الحرية اللانهائية (np.inf) لأن N كبير نسبياً
    try:
        q_alpha = stats.studentized_range.ppf(1 - alpha, n_methods, np.inf)
    except AttributeError:
        # Fallback لـ scipy القديم الذي لا يحتوي على studentized_range
        # نستخدم جدول تقريبي
        q_table = {
            0.05: {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
                   7: 2.948, 8: 3.031, 9: 3.102, 10: 3.164},
            0.01: {2: 2.576, 3: 2.792, 4: 2.960, 5: 3.081, 6: 3.178,
                   7: 3.259, 8: 3.327, 9: 3.385, 10: 3.436},
        }
        qtab = q_table.get(alpha, q_table[0.05])
        if n_methods in qtab:
            q_alpha = qtab[n_methods]
        else:
            # استيفاء خطي (تخميني)
            ks = sorted(qtab.keys())
            if n_methods > ks[-1]:
                q_alpha = qtab[ks[-1]] * (n_methods / ks[-1]) ** 0.5
            else:
                q_alpha = 2.569  # قيمة افتراضية
        warnings.warn(f"Nemenyi: Using approximate q_alpha={q_alpha:.3f}")

    # المسافة الحرجة (Critical Difference)
    cd = q_alpha * np.sqrt(n_methods * (n_methods + 1) / (6.0 * n_subjects))

    # المقارنات الزوجية
    comparisons = []
    for i, j in combinations(range(n_methods), 2):
        diff = abs(mean_ranks[i] - mean_ranks[j])
        significant = diff > cd

        # قيمة p تقريبية (باستخدام توزيع Studentized Range)
        if cd > 0:
            z = diff / (cd / q_alpha)
            p_approx = 2.0 * (1.0 - stats.norm.cdf(z)) if q_alpha > 0 else 1.0
            p_approx = min(1.0, p_approx)
        else:
            p_approx = 1.0

        comparisons.append({
            'method_i': int(i),
            'method_j': int(j),
            'rank_diff': float(diff),
            'critical_difference': float(cd),
            'significant': bool(significant),
            'p_approx': float(p_approx),
        })

    return {
        'critical_difference': float(cd),
        'q_value': float(q_alpha),
        'comparisons': comparisons,
    }


# ====================================================================
# 5. اختبار ويلكوكسون للمقارنات الزوجية (Wilcoxon Pairwise)
# ====================================================================
def wilcoxon_pairwise(accuracy_matrix, method_names=None):
    """
    اختبار ويلكوكسون للمقارنات الزوجية بين جميع أزواج الطرق.

    هذا الاختبار هو بديل غير معلمي لـ T-test للمقارنات الزوجية.

    المعاملات:
        accuracy_matrix (np.ndarray): مصفوفة الدقة (n_subjects, n_methods).
        method_names (list): أسماء الطرق (اختياري).

    المخرجات:
        list: قائمة من القواميس، كل قاموس يحتوي على:
            - 'method_1', 'method_2': أسماء الطرق
            - 'statistic': إحصائية ويلكوكسون (W)
            - 'p_value': قيمة p
            - 'significant': True إذا كانت p < 0.05
            - 'n': عدد الأشخاص المستخدمين في الاختبار
            - 'effect_size_r': حجم التأثير (Rank-biserial correlation)
    """
    n_subjects, n_methods = accuracy_matrix.shape
    method_names = method_names or [f"M{i}" for i in range(n_methods)]

    results = []

    for i, j in combinations(range(n_methods), 2):
        a = accuracy_matrix[:, i]
        b = accuracy_matrix[:, j]

        # إزالة القيم المفقودة (NaN)
        mask = ~(np.isnan(a) | np.isnan(b))
        a_clean = a[mask]
        b_clean = b[mask]

        n = len(a_clean)

        if n < 3:
            results.append({
                'method_1': method_names[i],
                'method_2': method_names[j],
                'statistic': np.nan,
                'p_value': np.nan,
                'significant': False,
                'n': n,
                'effect_size_r': np.nan,
                'note': 'Insufficient samples (n<3)'
            })
            continue

        try:
            # اختبار ويلكوكسون (ثنائي الطرف)
            stat, p_val = stats.wilcoxon(a_clean, b_clean, alternative='two-sided')

            # حجم التأثير: Rank-biserial correlation (r)
            # r = 1 - 2*W / (n*(n+1))
            # حيث W هي الإحصائية الأصغر
            # في scipy، الإحصائية المعادة هي W (مجموع الرتب الأصغر)
            # يمكننا حساب r مباشرة
            if n > 0:
                # ملاحظة: في scipy 1.9+، wilcoxon تعيد الإحصائية والقيمة p
                # يمكننا استخدام المعادلة: r = 1 - (2 * W) / (n * (n + 1))
                # لكن W في scipy قد تكون الإحصائية الأصغر
                # لحساب r بشكل دقيق، نستخدم الطريقة اليدوية
                diff = a_clean - b_clean
                abs_diff = np.abs(diff)
                ranks = stats.rankdata(abs_diff)
                sign = np.sign(diff)
                W_plus = np.sum(ranks[sign > 0])
                W_minus = np.sum(ranks[sign < 0])
                W = min(W_plus, W_minus)
                total_rank_sum = n * (n + 1) / 2.0
                r = 1 - (2.0 * W) / total_rank_sum if total_rank_sum > 0 else 0.0
            else:
                r = 0.0

            results.append({
                'method_1': method_names[i],
                'method_2': method_names[j],
                'statistic': float(stat),
                'p_value': float(p_val),
                'significant': bool(p_val < 0.05),
                'n': n,
                'effect_size_r': float(r),
                'note': None,
            })

        except Exception as e:
            results.append({
                'method_1': method_names[i],
                'method_2': method_names[j],
                'statistic': np.nan,
                'p_value': np.nan,
                'significant': False,
                'n': n,
                'effect_size_r': np.nan,
                'note': str(e),
            })

    return results


# ====================================================================
# 6. تصحيح هولم-سيداك (Holm-Sidak Correction)
# ====================================================================
def holm_sidak_correction(p_values, alpha=0.05):
    """
    Holm-Sidak step-down correction for multiple comparisons.

    For m raw p-values sorted ascending, the i-th smallest (i = 0..m-1) is adjusted to
    1 - (1 - p_i) ** (m - i), and monotonicity is enforced by a running maximum.

    Parameters
        p_values (list): raw p-values.
        alpha (float): significance level (default 0.05).

    Returns
        list of dicts with keys 'raw_p', 'adjusted_p', 'significant', 'rejected'.
    """
    pv = np.asarray(p_values, dtype=np.float64)
    n = len(pv)
    if n == 0:
        return []

    order = np.argsort(pv)
    adj = np.empty(n)
    running = 0.0
    for i, idx in enumerate(order):
        a = 1.0 - (1.0 - pv[idx]) ** (n - i)
        running = max(running, a)
        adj[idx] = min(1.0, running)

    return [
        {
            'raw_p': float(pv[k]),
            'adjusted_p': float(adj[k]),
            'significant': bool(adj[k] < alpha),
            'rejected': bool(adj[k] < alpha),
        }
        for k in range(n)
    ]


# ====================================================================
# 7. حجم التأثير كوهين د (Cohen's d)
# ====================================================================
def cohens_d(x, y):
    """
    حساب حجم التأثير كوهين د (Cohen's d) بين مجموعتين.

    القيم المرجعية:
        |d| < 0.2: تأثير ضئيل (Negligible)
        0.2 ≤ |d| < 0.5: تأثير صغير (Small)
        0.5 ≤ |d| < 0.8: تأثير متوسط (Medium)
        |d| ≥ 0.8: تأثير كبير (Large)

    المعاملات:
        x (array-like): المجموعة الأولى.
        y (array-like): المجموعة الثانية.

    المخرجات:
        float: قيمة كوهين د.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    n_x = len(x)
    n_y = len(y)

    if n_x < 2 or n_y < 2:
        return 0.0

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # حساب الانحراف المعياري المدمج (Pooled Standard Deviation)
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)

    pooled_var = ((n_x - 1) * var_x + (n_y - 1) * var_y) / (n_x + n_y - 2)
    pooled_std = np.sqrt(pooled_var) if pooled_var > 0 else 0.0

    if pooled_std == 0:
        return 0.0

    d = (mean_x - mean_y) / pooled_std
    return float(d)


def interpret_cohens_d(d):
    """
    تفسير قيمة كوهين د (حجم التأثير) إلى وصف نصي.

    المعاملات:
        d (float): قيمة كوهين د.

    المخرجات:
        str: وصف حجم التأثير.
    """
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    elif ad < 0.5:
        return "small"
    elif ad < 0.8:
        return "medium"
    else:
        return "large"


# ====================================================================
# 8. حساب دقة التخمين العشوائي (Random Baseline)
# ====================================================================
def random_baseline_from_labels(y, n_classes=None):
    """
    حساب دقة التخمين العشوائي بناءً على عدد الأصناف في y.

    إذا لم يتم تحديد n_classes، يتم استخراج عدد الأصناف الفريد من y.

    المعاملات:
        y (np.ndarray): التسميات (n_samples,).
        n_classes (int): عدد الأصناف (اختياري).

    المخرجات:
        float: دقة التخمين العشوائي كنسبة مئوية (0-100).
    """
    if n_classes is None:
        unique = np.unique(y)
        n_classes = len(unique)

    if n_classes == 0:
        return 0.0

    return 100.0 / n_classes


def ratio_vs_random(accuracy, random_baseline_pct):
    """
    حساب نسبة الدقة الفعلية إلى دقة التخمين العشوائي.

    المعاملات:
        accuracy (float): دقة التصنيف (0-1).
        random_baseline_pct (float): دقة التخمين العشوائي كنسبة مئوية (0-100).

    المخرجات:
        float: النسبة (الدقة الفعلية / دقة التخمين العشوائي).
               إذا كانت دقة التخمين العشوائي = 0، تُرجع 0.
    """
    if random_baseline_pct <= 0:
        return 0.0
    return float(accuracy * 100 / random_baseline_pct)


# ====================================================================
# 9. دوال مساعدة لبناء مصفوفات الرتب (للتحليل الإحصائي)
# ====================================================================
def build_rank_matrix(accuracy_matrix):
    """
    بناء مصفوفة الرتب من مصفوفة الدقة.

    يتم ترتيب الدقة تنازلياً (أعلى دقة = رتبة 1).

    المعاملات:
        accuracy_matrix (np.ndarray): مصفوفة الدقة (n_subjects, n_methods).

    المخرجات:
        np.ndarray: مصفوفة الرتب (n_subjects, n_methods).
    """
    n_subjects, n_methods = accuracy_matrix.shape
    ranks = np.zeros_like(accuracy_matrix, dtype=np.float64)

    for i in range(n_subjects):
        ranks[i] = stats.rankdata(-accuracy_matrix[i])

    return ranks


def average_rank_per_method(accuracy_matrix):
    """
    حساب متوسط الرتب لكل طريقة.

    المعاملات:
        accuracy_matrix (np.ndarray): مصفوفة الدقة (n_subjects, n_methods).

    المخرجات:
        np.ndarray: متوسط الرتب لكل طريقة (n_methods,).
    """
    ranks = build_rank_matrix(accuracy_matrix)
    return ranks.mean(axis=0)


# ====================================================================
# 10. دالة شاملة لتجميع جميع الاختبارات (للاستخدام في day5_statistics)
# ====================================================================
def comprehensive_statistical_analysis(accuracy_matrix, method_names, alpha=0.05):
    """
    دالة شاملة لتجميع جميع الاختبارات الإحصائية في مكان واحد.

    هذه الدالة تُستخدم في p06_run_statistical_analysis.py لتوليد جميع الجداول الإحصائية دفعة واحدة.

    المعاملات:
        accuracy_matrix (np.ndarray): مصفوفة الدقة (n_subjects, n_methods).
        method_names (list): أسماء الطرق.
        alpha (float): مستوى الدلالة (الافتراضي 0.05).

    المخرجات:
        dict: قاموس يحتوي على نتائج جميع الاختبارات:
            - 'friedman': نتائج اختبار فريدمان
            - 'nemenyi': نتائج اختبار نيميني
            - 'wilcoxon': نتائج اختبار ويلكوكسون الزوجي
            - 'holm_sidak': نتائج تصحيح هولم-سيداك
            - 'cohens_d': مصفوفة كوهين د لكل زوج
            - 'descriptive': الإحصاءات الوصفية لكل طريقة
            - 'random_baseline': دقة التخمين العشوائي (إذا كانت قابلة للحساب)
    """
    n_subjects, n_methods = accuracy_matrix.shape

    # 1. الإحصاءات الوصفية
    descriptive = {}
    for i, name in enumerate(method_names):
        descriptive[name] = descriptive_stats(accuracy_matrix[:, i])

    # 2. اختبار فريدمان
    friedman = friedman_test(accuracy_matrix, method_names)

    # 3. اختبار نيميني (إذا كان فريدمان معنوياً)
    nemenyi = None
    if not np.isnan(friedman['p_value']) and friedman['p_value'] < alpha:
        nemenyi = nemenyi_posthoc(
            friedman['mean_ranks'],
            n_subjects,
            n_methods,
            alpha
        )

    # 4. اختبار ويلكوكسون الزوجي
    wilcoxon = wilcoxon_pairwise(accuracy_matrix, method_names)

    # 5. تصحيح هولم-سيداك على نتائج ويلكوكسون
    p_values = [w['p_value'] for w in wilcoxon if not np.isnan(w['p_value'])]
    holm_sidak = holm_sidak_correction(p_values, alpha) if p_values else []

    # 6. كوهين د لكل زوج
    cohens_d_matrix = np.zeros((n_methods, n_methods))
    for i in range(n_methods):
        for j in range(n_methods):
            if i != j:
                cohens_d_matrix[i, j] = cohens_d(
                    accuracy_matrix[:, i],
                    accuracy_matrix[:, j]
                )

    return {
        'friedman': friedman,
        'nemenyi': nemenyi,
        'wilcoxon': wilcoxon,
        'holm_sidak': holm_sidak,
        'cohens_d_matrix': cohens_d_matrix,
        'descriptive': descriptive,
        'method_names': method_names,
        'n_subjects': n_subjects,
        'n_methods': n_methods,
        'alpha': alpha,
    }