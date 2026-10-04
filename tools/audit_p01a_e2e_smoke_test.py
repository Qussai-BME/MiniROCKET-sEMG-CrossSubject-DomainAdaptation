"""
audit_p01a_e2e_smoke_test.py — end-to-end smoke test of the patched pipeline
========================================================================
يبني "قاعدة بيانات" اصطناعية كاملة (عدة أشخاص، بصيغة .mat حقيقية) ويُشغّل
p01a.run_single_fold الحقيقية (بدون أي تعديل/محاكاة) عبر الطرق الخمس
(raw, centering, coral, tca, sa) وكلا المستخرجين (custom_ppv، canonical)،
للتحقق من أن كل السلك (data_loader -> subject_split -> feature_extractors
-> domain_adaptation -> metrics) يعمل معاً دون أخطاء وينتج نتائج منطقية:

  - الدقة في [0، 1]
  - balanced_accuracy وmacro_f1 موجودتان ومحدودتان
  - n_calibration_samples_used = 0 لـ raw/centering، > 0 لـ coral/tca/sa
  - evaluation_paradigm صحيحة لكل طريقة
  - لا استثناءات غير متوقعة

هذا لا يستبدل التشغيل الحقيقي على NinaPro (الدقة هنا بلا معنى إحصائياً —
البيانات عشوائية بالكامل) لكنه يثبت أن الأنابيب المُصلَحة تعمل ميكانيكياً
قبل إنفاق ساعات حوسبة حقيقية.
"""
import os
import sys
import tempfile
import shutil
import numpy as np
import scipy.io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from data_loader import load_ninapro_stream
from p01a_run_main_loso_benchmark import run_single_fold


def _build_subject_mat(path, n_movements, n_reps, n_channels, fs, seed,
                       class_signal_gain=1.0):
    rng = np.random.RandomState(seed)
    active_len = fs // 2
    rest_len   = fs // 4
    emg_chunks, stim_chunks, rep_chunks = [], [], []

    def add_rest():
        emg_chunks.append(rng.randn(rest_len, n_channels).astype(np.float32) * 0.05)
        stim_chunks.append(np.zeros(rest_len, dtype=np.int32))
        rep_chunks.append(np.zeros(rest_len, dtype=np.int32))

    add_rest()
    for r in range(1, n_reps + 1):
        for m in range(1, n_movements + 1):
            # كل صنف له "بصمة" ضعيفة في القناة m%n_channels لإعطاء بعض
            # البنية القابلة للتعلم (وليس ضجيجاً بحتاً) دون الحاجة لبيانات حقيقية
            base = rng.randn(active_len, n_channels).astype(np.float32)
            base[:, m % n_channels] += class_signal_gain
            emg_chunks.append(base)
            stim_chunks.append(np.full(active_len, m, dtype=np.int32))
            rep_chunks.append(np.full(active_len, r, dtype=np.int32))
            add_rest()

    emg  = np.concatenate(emg_chunks, axis=0)
    stim = np.concatenate(stim_chunks, axis=0)
    reps = np.concatenate(rep_chunks, axis=0)
    scipy.io.savemat(path, {
        "emg": emg.astype(np.float64),
        "restimulus": stim.reshape(-1, 1).astype(np.float64),
        "repetition": reps.reshape(-1, 1).astype(np.float64),
        "sampling_frequency": np.array([[fs]], dtype=np.float64),
    })


class _FakeCfg:
    pass


def main():
    tmpdir = tempfile.mkdtemp(prefix="audit_p01a_e2e_")
    try:
        n_movements, n_reps, n_channels, fs = 5, 6, 8, 2000
        n_subjects = 4

        for sid in range(1, n_subjects + 1):
            path = os.path.join(tmpdir, f"S{sid}_E1_A1.mat")
            _build_subject_mat(path, n_movements, n_reps, n_channels, fs,
                              seed=sid, class_signal_gain=3.0)

        fake_cfg = _FakeCfg()
        fake_cfg.DB_PATHS = {"synthtest": tmpdir}
        fake_cfg.DB_META = {
            "synthtest": {
                "n_subjects": n_subjects,
                "n_movements": n_movements,
                "n_channels": n_channels,
                "sampling_rate": fs,
                "subject_ids": list(range(1, n_subjects + 1)),
                "movement_ids": list(range(1, n_movements + 1)),
                "exclude_rest": True,
                "display_name": "Synthetic E2E Test DB",
                "exercise_e1_only": False,
                "grouped_classes": None,
            }
        }

        # نحمّل كل الأشخاص مرة واحدة في الذاكرة (قاعدة بيانات صغيرة جداً)
        all_subjects = {
            s["subject_id"]: s
            for s in load_ninapro_stream("synthtest", fake_cfg,
                                         window_ms=200, overlap=0.5)
        }
        assert len(all_subjects) == n_subjects
        test_sid = 1
        test_data = all_subjects[test_sid]

        def make_train_gen():
            for sid, s in all_subjects.items():
                if sid != test_sid:
                    yield s

        n_classes = n_movements
        results = {}

        for extractor in ["custom_ppv", "canonical"]:
            for method in config.METHODS:
                print(f"\n--- extractor={extractor}  method={method} ---",
                      flush=True)
                res = run_single_fold(
                    make_train_gen(), test_data, method,
                    num_kernels=500,          # صغير جداً، للسرعة فقط
                    n_classes=n_classes,
                    max_train_samples=5000,
                    per_subject_limit=2000,
                    seed=42,
                    feature_extractor=extractor,
                    calibration_fraction=0.5,
                )
                key = f"{extractor}/{method}"
                results[key] = res
                assert 'error' not in res, f"{key} FAILED: {res.get('error')}"
                assert 0.0 <= res['accuracy'] <= 1.0
                assert 0.0 <= res['balanced_accuracy'] <= 1.0
                assert 0.0 <= res['macro_f1'] <= 1.0
                expected_paradigm = config.EVALUATION_PARADIGM[method]
                assert res['evaluation_paradigm'] == expected_paradigm
                if method in config.TRANSDUCTIVE_METHODS:
                    assert res['n_calibration_samples_used'] > 0, \
                        f"{key}: transductive method used 0 calibration samples"
                else:
                    assert res['n_calibration_samples_used'] == 0, \
                        f"{key}: inductive method must not touch calibration data"
                print(f"    OK: acc={res['accuracy']:.3f} "
                      f"bal_acc={res['balanced_accuracy']:.3f} "
                      f"macro_f1={res['macro_f1']:.3f} "
                      f"paradigm={res['evaluation_paradigm']} "
                      f"calib_used={res['n_calibration_samples_used']}")

        print("\n" + "=" * 72)
        print("ALL 10 (extractor x method) COMBINATIONS RAN SUCCESSFULLY "
              "THROUGH THE REAL p01a.run_single_fold().")
        print("=" * 72)
        return 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
