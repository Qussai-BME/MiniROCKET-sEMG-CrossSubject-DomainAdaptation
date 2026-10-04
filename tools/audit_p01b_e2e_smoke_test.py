"""
audit_p01b_e2e_smoke_test.py — verifies the repetition-level within-subject split end to end
"""
import os, sys, tempfile, shutil
import numpy as np
import scipy.io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import load_ninapro_stream
from p01b_run_within_subject_baseline import run_within_subject_fold


def _build_subject_mat(path, n_movements, n_reps, n_channels, fs, seed):
    rng = np.random.RandomState(seed)
    active_len, rest_len = fs // 2, fs // 4
    emg_chunks, stim_chunks, rep_chunks = [], [], []

    def add_rest():
        emg_chunks.append(rng.randn(rest_len, n_channels).astype(np.float32) * 0.05)
        stim_chunks.append(np.zeros(rest_len, dtype=np.int32))
        rep_chunks.append(np.zeros(rest_len, dtype=np.int32))

    add_rest()
    for r in range(1, n_reps + 1):
        for m in range(1, n_movements + 1):
            base = rng.randn(active_len, n_channels).astype(np.float32)
            base[:, m % n_channels] += 3.0
            emg_chunks.append(base)
            stim_chunks.append(np.full(active_len, m, dtype=np.int32))
            rep_chunks.append(np.full(active_len, r, dtype=np.int32))
            add_rest()
    scipy.io.savemat(path, {
        "emg": np.concatenate(emg_chunks).astype(np.float64),
        "restimulus": np.concatenate(stim_chunks).reshape(-1, 1).astype(np.float64),
        "repetition": np.concatenate(rep_chunks).reshape(-1, 1).astype(np.float64),
        "sampling_frequency": np.array([[fs]], dtype=np.float64),
    })


class _FakeCfg:
    pass


def main():
    tmpdir = tempfile.mkdtemp(prefix="audit_p01b_e2e_")
    try:
        n_movements, n_reps, n_channels, fs = 5, 6, 8, 2000
        path = os.path.join(tmpdir, "S1_E1_A1.mat")
        _build_subject_mat(path, n_movements, n_reps, n_channels, fs, seed=1)

        cfg = _FakeCfg()
        cfg.DB_PATHS = {"synthtest": tmpdir}
        cfg.DB_META = {"synthtest": {
            "n_subjects": 1, "n_movements": n_movements, "n_channels": n_channels,
            "sampling_rate": fs, "subject_ids": [1],
            "movement_ids": list(range(1, n_movements + 1)),
            "exclude_rest": True, "display_name": "test",
            "exercise_e1_only": False, "grouped_classes": None,
        }}

        subj = next(load_ninapro_stream("synthtest", cfg, window_ms=200, overlap=0.5))
        X, y, rep_id = subj["X"], subj["y"], subj["rep_id"]
        print(f"Loaded: X={X.shape} y={y.shape} rep_id unique={np.unique(rep_id)}")

        fold = run_within_subject_fold(X, y, rep_id, num_kernels=500,
                                       feature_extractor="custom_ppv",
                                       test_fraction=0.2, seed=42)
        print(fold)
        assert "error" not in fold, fold.get("error")
        assert fold["split_unit"] == "repetition"
        assert 0.0 <= fold["accuracy"] <= 1.0
        assert 0.0 <= fold["balanced_accuracy"] <= 1.0
        assert fold["n_train_repetitions"] > 0 and fold["n_test_repetitions"] > 0

        # التحقق الصحيح: (class, repetition) هو المقطع الخام غير القابل
        # للتقسيم الجزئي — وليس رقم التكرار وحده عبر كل الأصناف (كل صنف
        # يملك تقسيمه الطبقي المستقل، لذا "تكرار 3" قد يكون train لصنف
        # وtest لصنف آخر دون أي تسريب، لأنهما مقطعان خامان مختلفان تماماً).
        from subject_split import split_by_repetition
        mask_train, mask_test, _ = split_by_repetition(
            y, rep_id, held_out_fraction=0.2, seed=42,
            min_reps_per_class_each_side=1)
        leak = 0
        for cls in np.unique(y):
            m = (y == cls)
            reps_tr = set(rep_id[m & mask_train].tolist())
            reps_te = set(rep_id[m & mask_test].tolist())
            if reps_tr & reps_te:
                leak += 1
        assert leak == 0, f"{leak} class(es) have a (class,repetition) leaking across train/test"
        print("\nPASS: p01b repetition-level within-subject split runs "
              "end-to-end through the real code with no window-level leakage.")
        return 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
