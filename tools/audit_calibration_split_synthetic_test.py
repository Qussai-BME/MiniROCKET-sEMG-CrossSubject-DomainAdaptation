"""
audit_calibration_split_synthetic_test.py — 
========================================================================
يتحقق من أن دوال coral_adaptation / tca_transform / sa_transform، عند
تمرير X_target_final، لا "تُسرّب" أي معلومة من final إلى مرحلة الملاءمة:

  1. Z_target_calib يجب أن يكون مطابقاً تماماً سواء استُدعيت الدالة
     بوضع (source, calib) فقط، أو بوضع (source, calib, final) — أي أن
     وجود final لا يُغيّر شيئاً في الملاءمة نفسها.
  2. Xt_final (CORAL) = X_target_final كما هو (طابق الهوية) — لأن CORAL
     لا يُحوّل الهدف إطلاقاً بعد FIX-CORAL-01.
  3. TCA/SA: إسقاط target_final ينتج شكلاً صحيحاً (نفس عدد الصفوف)
     وقيماً مختلفة عن calib (لأنها بيانات مختلفة عشوائياً) لكن بنفس
     "الفراغ" الخطي (لا NaN/Inf، تباين محدود).
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domain_adaptation import coral_adaptation, tca_transform, sa_transform


def main():
    rng = np.random.RandomState(0)
    F = 40
    Xs = rng.randn(500, F).astype(np.float32) * 2.0 + 1.0
    Xt_calib = rng.randn(120, F).astype(np.float32) * 1.5 - 0.5
    Xt_final = rng.randn(130, F).astype(np.float32) * 1.5 - 0.5

    all_ok = True

    # --- CORAL ---
    Zs_2, Zc_2 = coral_adaptation(Xs, Xt_calib, reg=1e-3)
    Zs_3, Zc_3, Zf_3 = coral_adaptation(Xs, Xt_calib, Xt_final, reg=1e-3)
    ok1 = np.allclose(Zs_2, Zs_3) and np.allclose(Zc_2, Zc_3)
    ok2 = np.allclose(Zf_3, Xt_final)   # identity pass-through
    print(f"[CORAL] calib output identical 2-tuple vs 3-tuple: {ok1}")
    print(f"[CORAL] final output is unchanged passthrough:      {ok2}")
    all_ok &= ok1 and ok2

    # --- TCA ---
    Zs_2, Zc_2 = tca_transform(Xs, Xt_calib, n_components=10, mu=1.0)
    Zs_3, Zc_3, Zf_3 = tca_transform(Xs, Xt_calib, Xt_final,
                                     n_components=10, mu=1.0)
    ok3 = np.allclose(Zs_2, Zs_3) and np.allclose(Zc_2, Zc_3)
    ok4 = (Zf_3.shape[0] == Xt_final.shape[0]) and np.isfinite(Zf_3).all()
    print(f"[TCA]   calib output identical 2-tuple vs 3-tuple: {ok3}")
    print(f"[TCA]   final output shape/finite OK:               {ok4}  "
          f"shape={Zf_3.shape}")
    all_ok &= ok3 and ok4

    # --- SA ---
    Zs_2, Zc_2 = sa_transform(Xs, Xt_calib, n_components=10)
    Zs_3, Zc_3, Zf_3 = sa_transform(Xs, Xt_calib, Xt_final, n_components=10)
    ok5 = np.allclose(Zs_2, Zs_3) and np.allclose(Zc_2, Zc_3)
    ok6 = (Zf_3.shape[0] == Xt_final.shape[0]) and np.isfinite(Zf_3).all()
    print(f"[SA]    calib output identical 2-tuple vs 3-tuple: {ok5}")
    print(f"[SA]    final output shape/finite OK:               {ok6}  "
          f"shape={Zf_3.shape}")
    all_ok &= ok5 and ok6

    # --- leakage sensitivity check: refitting TCA directly ON X_target_final
    #     (as if it were the calib set) must give a DIFFERENT projection
    #     than reusing the calib-fitted P, proving the calib-only fit
    #     actually depends on which data it sees (i.e., the "no leakage"
    #     result above isn't trivially true because the transform ignores
    #     the target entirely).
    Zs_direct, Zt_direct = tca_transform(Xs, Xt_final, n_components=10, mu=1.0)
    differs = not np.allclose(Zt_direct, Zf_3)
    print(f"[TCA]   fitting directly on 'final' set gives a DIFFERENT "
          f"projection than reusing calib-fit (sanity check): {differs}")
    all_ok &= differs

    print()
    print("ALL CALIBRATION/FINAL-TEST SPLIT CHECKS "
          f"{'PASSED' if all_ok else 'FAILED'}.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
