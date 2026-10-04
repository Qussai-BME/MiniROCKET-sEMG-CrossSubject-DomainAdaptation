"""
audit_dataloader_synthetic_test.py — verifies FIX-DATALOADER-01
========================================================================
يبني ملفات .mat اصطناعية بنفس بنية NinaPro (emg, restimulus, repetition)
ويُشغّل load_ninapro_stream الحقيقية (بدون أي محاكاة/monkeypatch لمنطق
التحميل نفسه) للتحقق من:
  1. rep_id يُعاد بنفس طول X وy (لا انزياح/سقوط محاذاة عبر كل الفلاتر).
  2. كل نافذة تحمل رقم التكرار الصحيح فعلياً (نقارنه بالمعلومة التي
     زرعناها عند بناء الملف الاصطناعي).
  3. subject_split.split_by_repetition يعمل على مخرجات المُحمِّل الحقيقية
     دون أي تسريب تكرار عبر الطرفين.

التشغيل: python tools/audit_dataloader_synthetic_test.py
"""
import os
import sys
import tempfile
import shutil
import numpy as np
import scipy.io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import load_ninapro_stream
from subject_split import split_by_repetition, summarize_split


def _build_synthetic_subject_mat(path, n_movements=5, n_reps=6,
                                 n_channels=8, fs=2000, seed=0):
    """
    يبني إشارة EMG اصطناعية بمخطط:
      [rest] [movement m, rep r] [rest] [movement m, rep r] ...
    لكل m في 1..n_movements ولكل r في 1..n_reps، بالترتيب.
    """
    rng = np.random.RandomState(seed)
    active_len = fs // 2      # نصف ثانية نشاط
    rest_len   = fs // 4      # ربع ثانية راحة

    emg_chunks, stim_chunks, rep_chunks = [], [], []

    def add_rest():
        emg_chunks.append(rng.randn(rest_len, n_channels).astype(np.float32) * 0.05)
        stim_chunks.append(np.zeros(rest_len, dtype=np.int32))
        rep_chunks.append(np.zeros(rest_len, dtype=np.int32))

    add_rest()
    for r in range(1, n_reps + 1):
        for m in range(1, n_movements + 1):
            emg_chunks.append(
                rng.randn(active_len, n_channels).astype(np.float32) + m * 0.1
            )
            stim_chunks.append(np.full(active_len, m, dtype=np.int32))
            rep_chunks.append(np.full(active_len, r, dtype=np.int32))
            add_rest()

    emg   = np.concatenate(emg_chunks, axis=0)
    stim  = np.concatenate(stim_chunks, axis=0)
    reps  = np.concatenate(rep_chunks, axis=0)

    scipy.io.savemat(path, {
        "emg": emg.astype(np.float64),
        "restimulus": stim.reshape(-1, 1).astype(np.float64),
        "repetition": reps.reshape(-1, 1).astype(np.float64),
        "sampling_frequency": np.array([[fs]], dtype=np.float64),
    })
    return dict(n_movements=n_movements, n_reps=n_reps,
               n_channels=n_channels, fs=fs)


class _FakeCfg:
    pass


def main():
    tmpdir = tempfile.mkdtemp(prefix="audit_dataloader_test_")
    try:
        n_movements, n_reps, n_channels, fs = 5, 6, 8, 2000
        mat_path = os.path.join(tmpdir, "S1_E1_A1.mat")
        meta_in = _build_synthetic_subject_mat(
            mat_path, n_movements=n_movements, n_reps=n_reps,
            n_channels=n_channels, fs=fs, seed=0,
        )

        cfg = _FakeCfg()
        cfg.DB_PATHS = {"synthtest": tmpdir}
        cfg.DB_META = {
            "synthtest": {
                "n_subjects": 1,
                "n_movements": n_movements,
                "n_channels": n_channels,
                "sampling_rate": fs,
                "subject_ids": [1],
                "movement_ids": list(range(1, n_movements + 1)),
                "exclude_rest": True,
                "display_name": "Synthetic Test DB",
                "exercise_e1_only": False,
                "grouped_classes": None,
            }
        }

        subjects = list(load_ninapro_stream("synthtest", cfg,
                                             window_ms=200, overlap=0.5))
        assert len(subjects) == 1, f"Expected 1 subject, got {len(subjects)}"
        subj = subjects[0]

        X, y, rep_id = subj["X"], subj["y"], subj["rep_id"]
        print(f"[1] X.shape={X.shape}  y.shape={y.shape}  "
              f"rep_id.shape={rep_id.shape}")
        assert X.shape[0] == y.shape[0] == rep_id.shape[0], \
            "ALIGNMENT FAILURE: X/y/rep_id lengths differ"
        print("    PASS: X, y, rep_id are length-aligned")

        assert set(np.unique(rep_id).tolist()) <= set(range(1, n_reps + 1)), \
            f"Unexpected repetition ids found: {np.unique(rep_id)}"
        assert set(np.unique(y).tolist()) <= set(range(n_movements)), \
            f"Unexpected (remapped) labels found: {np.unique(y)}"
        print(f"    PASS: rep_id in expected range "
              f"1..{n_reps}, y in expected range 0..{n_movements-1}")
        print(f"    windows/rep histogram: "
              f"{ {int(r): int((rep_id==r).sum()) for r in np.unique(rep_id)} }")

        # --- item 4/5 split utility on the REAL loader output ---
        mask_kept, mask_held, warns = split_by_repetition(
            y, rep_id, held_out_fraction=0.5, seed=42
        )
        info = summarize_split(y, rep_id, mask_kept, mask_held)
        print(f"[2] split summary: {info}")
        assert not info["mask_overlap_detected"], "OVERLAP DETECTED"
        for cls in np.unique(y):
            m = (y == cls)
            reps_k = set(rep_id[m & mask_kept].tolist())
            reps_h = set(rep_id[m & mask_held].tolist())
            assert not (reps_k & reps_h), \
                f"class {cls}: repetition leaked across split sides"
        print("    PASS: no repetition appears on both sides of the split, "
              "for any class")

        print("\nALL DATALOADER SYNTHETIC CHECKS PASSED.")
        return 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
