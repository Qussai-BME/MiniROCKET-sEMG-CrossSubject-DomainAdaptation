"""
audit_coral_synthetic_test.py
==============================================================================
اختبار CORAL على بيانات غاوسية اصطناعية — تدقيق المشرف، البند 2 (أعلى أولوية).

الغرض: التحقق رياضيًا، على بيانات مُصطنعة (وليس NinaPro)، من أن CORAL بعد
الإصلاح (FIX-CORAL-01) يُقلّل فعليًا مسافة Frobenius بين تغايرَي المصدر
والهدف — قبل إعادة تشغيل أي تجربة حقيقية على NinaPro، بما يتوافق مع طلب
المشرف: "CORAL should pass a synthetic Gaussian covariance test where
adaptation demonstrably decreases source-target covariance distance."

هذا الاختبار يقارن أيضًا صراحة بين:
  (أ) السلوك القديم (المعطوب): الهدف يُعاد "تلوينه" بمصفوفة Cholesky
      الخاصة بتغايره هو (البق الذي رصده التدقيق)
  (ب) السلوك بعد الإصلاح: الهدف يبقى دون أي تحويل

لإثبات أن (أ) كان يزيد التباعد أحيانًا بينما (ب) يُقلّله دائمًا، تمامًا
كما ورد في تدقيق المشرف.

التشغيل:
    python tools/audit_coral_synthetic_test.py
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domain_adaptation import coral_adaptation, covariance_frobenius


def _random_spd_matrix(dim, scale, rng):
    """مصفوفة تغاير عشوائية موجبة التحديد قطعًا (SPD) بحجم dim×dim."""
    A = rng.standard_normal((dim, dim)) * scale
    return A @ A.T + np.eye(dim) * 1e-2


def _sample_gaussian(n, mean, cov, rng):
    return rng.multivariate_normal(mean=mean, cov=cov, size=n)


def _old_buggy_coral(X_source, X_target, reg=1e-3):
    """
    إعادة إنتاج السلوك القديم (قبل FIX-CORAL-01) لأغراض المقارنة فقط:
    الهدف كان يُعاد تحويله بـ (X_target - mean_t) @ Lt + mean_t، أي يُعاد
    "تلوينه" بمصفوفة Cholesky الخاصة بتغايره هو نفسه.
    """
    from scipy.linalg import cholesky, solve, LinAlgError

    X_source = np.asarray(X_source, dtype=np.float64)
    X_target = np.asarray(X_target, dtype=np.float64)
    n_s, F = X_source.shape

    mean_s = X_source.mean(axis=0)
    mean_t = X_target.mean(axis=0)
    Xs_c = X_source - mean_s
    Xt_c = X_target - mean_t

    Cs = (Xs_c.T @ Xs_c) / max(n_s - 1, 1)
    Ct = (Xt_c.T @ Xt_c) / max(X_target.shape[0] - 1, 1)
    Cs_reg = Cs + reg * np.eye(F)
    Ct_reg = Ct + reg * np.eye(F)

    Ls = cholesky(Cs_reg, lower=True)
    Lt = cholesky(Ct_reg, lower=True)
    Ls_inv = solve(Ls, np.eye(F))
    W = Ls_inv @ Lt

    Xs_aligned = Xs_c @ W + mean_t
    # <-- البق: إعادة تلوين الهدف بمصفوفة تغايره هو نفسه
    Xt_aligned = (X_target - mean_t) @ Lt + mean_t
    return Xs_aligned.astype(np.float32), Xt_aligned.astype(np.float32)


def run_case(name, dim, n_source, n_target, mean_shift, cov_scale_s,
             cov_scale_t, seed):
    rng = np.random.RandomState(seed)
    mean_s = np.zeros(dim)
    mean_t = np.ones(dim) * mean_shift

    Cs = _random_spd_matrix(dim, cov_scale_s, rng)
    Ct = _random_spd_matrix(dim, cov_scale_t, rng)

    X_source = _sample_gaussian(n_source, mean_s, Cs, rng)
    X_target = _sample_gaussian(n_target, mean_t, Ct, rng)

    dist_before = covariance_frobenius(X_source, X_target)

    # السلوك بعد الإصلاح (الدالة الحقيقية في domain_adaptation.py)
    Xs_fixed, Xt_fixed = coral_adaptation(X_source, X_target)
    dist_after_fixed = covariance_frobenius(Xs_fixed, Xt_fixed)

    # السلوك القديم المعطوب، للمقارنة فقط
    Xs_old, Xt_old = _old_buggy_coral(X_source, X_target)
    dist_after_old = covariance_frobenius(Xs_old, Xt_old)

    reduction_fixed = (dist_before - dist_after_fixed) / dist_before * 100
    reduction_old = (dist_before - dist_after_old) / dist_before * 100

    passed = dist_after_fixed < dist_before

    print(f"\n[{name}] dim={dim}, n_source={n_source}, n_target={n_target}")
    print(f"  Frobenius distance BEFORE CORAL         : {dist_before:10.4f}")
    print(f"  Frobenius distance AFTER  (old/buggy)    : {dist_after_old:10.4f}"
          f"   ({reduction_old:+.1f}% change)")
    print(f"  Frobenius distance AFTER  (fixed)        : {dist_after_fixed:10.4f}"
          f"   ({reduction_fixed:+.1f}% change)")
    print(f"  PASS (fixed distance decreased)?         : {passed}")

    return passed, dist_before, dist_after_fixed, dist_after_old


def main():
    print("=" * 78)
    print("  FIX-CORAL-01 — Synthetic Gaussian covariance test")
    print("=" * 78)

    cases = [
        dict(name="Small, well-conditioned (sanity)", dim=20,
             n_source=2000, n_target=500, mean_shift=2.0,
             cov_scale_s=1.0, cov_scale_t=1.5, seed=1),
        dict(name="Medium, moderate shift", dim=200,
             n_source=4000, n_target=1000, mean_shift=1.0,
             cov_scale_s=1.0, cov_scale_t=2.5, seed=2),
        dict(name="High-dim, PPV-like scale (10k features)", dim=500,
             n_source=3000, n_target=800, mean_shift=0.5,
             cov_scale_s=0.5, cov_scale_t=1.2, seed=3),
        dict(name="Large covariance mismatch (stress case)", dim=100,
             n_source=3000, n_target=900, mean_shift=3.0,
             cov_scale_s=0.3, cov_scale_t=3.0, seed=4),
    ]

    results = []
    for case in cases:
        results.append(run_case(**case))

    all_passed = all(r[0] for r in results)

    print("\n" + "=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    for (passed, before, after_fixed, after_old), case in zip(results, cases):
        flag = "PASS" if passed else "FAIL"
        print(f"  [{flag}] {case['name']}: "
              f"{before:.2f} -> {after_fixed:.2f} (fixed) "
              f"| {before:.2f} -> {after_old:.2f} (old/buggy)")

    print()
    if all_passed:
        print("  ALL CASES PASSED: fixed CORAL demonstrably decreases "
              "source-target covariance distance on synthetic Gaussian data.")
    else:
        print("  FAILURE: at least one case did not show a distance "
              "reduction after the fix. Do not proceed to NinaPro rerun.")
    print("=" * 78)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
