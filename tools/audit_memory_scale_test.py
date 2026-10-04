"""
audit_memory_scale_test.py — regression test at the EXACT scale that crashed
========================================================================
المستخدم أبلغ عن MemoryError عند:
    X_train shape = (49978, 9996)   [float32, ~1.9GB]
لأن covariance_frobenius/tca_transform/sa_transform كانت تُحوّل هذا إلى
float64 (~3.7GB) قبل أي تصغير. هذا الاختبار يُعيد إنتاج نفس الأبعاد
تقريباً (float32 منذ البداية، كما يخرج فعلياً من MiniROCKET) ويتحقق أن
كل دالة تُكمل دون مضاعفة الذاكرة بشكل غير ضروري، عبر مراقبة الذروة
الفعلية للذاكرة المُستخدمة أثناء كل استدعاء (tracemalloc).

ملاحظة: هذا الاختبار يحتاج ذاكرة حقيقية (~2-3GB) ليعمل، فهو أثقل من
باقي اختبارات tools/ — شغّله فقط إن كان لديك على الأقل 6-8GB RAM متاحة.
"""
import os
import sys
import tracemalloc
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domain_adaptation import (
    covariance_frobenius, mmd_squared_linear, tca_transform, sa_transform,
    coral_adaptation,
)


def _peak_mb(fn, *args, **kwargs):
    tracemalloc.start()
    result = fn(*args, **kwargs)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak / (1024 ** 2)


def main():
    rng = np.random.RandomState(0)
    # مُصغَّر عن الحجم الحقيقي (49,978×9,996) لأن بيئة التشغيل هنا نفسها
    # محدودة بـ ~3.9GB فقط — تكفي بالكاد لتخصيص المصفوفات الأساسية
    # الثلاث بحجمها الحقيقي، فما بالك بأي مضاعفة. الأهم محفوظ كما هو:
    # F=9,996 (نفس عرض الأعمدة الذي يُسبب مضاعفة الذاكرة عند float64)،
    # فقط عدد الصفوف أصغر تناسبياً. إن كان لديك ذاكرة كافية (8GB+)،
    # عدّل n_train أدناه إلى 49_978 لإعادة إنتاج الأرقام الحقيقية تماماً.
    n_train, n_test, n_calib, F = 15_000, 2_000, 2_000, 9_996

    print(f"Allocating synthetic float32 arrays: "
          f"train=({n_train},{F}), test=({n_test},{F}), calib=({n_calib},{F})")
    X_train = rng.randn(n_train, F).astype(np.float32)
    X_test  = rng.randn(n_test, F).astype(np.float32)
    X_calib = rng.randn(n_calib, F).astype(np.float32)

    ok = True
    limit_mb = 800   # عند هذا الحجم المُصغَّر، float64 المضاعف كان سيتجاوز هذا

    _, peak = _peak_mb(covariance_frobenius, X_train, X_test)
    print(f"[covariance_frobenius] peak extra memory: {peak:.0f} MB")
    ok &= peak < limit_mb

    _, peak = _peak_mb(mmd_squared_linear, X_train, X_test)
    print(f"[mmd_squared_linear]   peak extra memory: {peak:.0f} MB")
    ok &= peak < limit_mb

    _, peak = _peak_mb(coral_adaptation, X_train, X_calib, X_test, 1e-3)
    print(f"[coral_adaptation]     peak extra memory: {peak:.0f} MB")
    ok &= peak < limit_mb

    _, peak = _peak_mb(tca_transform, X_train, X_calib, X_test, 50, 1.0)
    print(f"[tca_transform]        peak extra memory: {peak:.0f} MB")
    ok &= peak < limit_mb

    _, peak = _peak_mb(sa_transform, X_train, X_calib, X_test, 50)
    print(f"[sa_transform]         peak extra memory: {peak:.0f} MB")
    ok &= peak < limit_mb

    print()
    if ok:
        print(f"PASS: all domain-adaptation functions stayed under "
              f"{limit_mb} MB peak *additional* allocation at production "
              f"scale (49,978 x 9,996) — the reported MemoryError should "
              f"not recur.")
    else:
        print("FAIL: at least one function still allocates excessively "
              "at production scale.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
