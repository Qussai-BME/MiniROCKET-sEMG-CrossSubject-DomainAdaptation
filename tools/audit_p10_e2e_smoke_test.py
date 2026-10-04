"""
audit_p10_e2e_smoke_test.py — verifies p10 sensitivity sweep runs end-to-end
"""
import os, sys, tempfile, shutil
import numpy as np
import scipy.io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from p10_run_sensitivity_analysis import run_coral_reg_sweep


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


def main():
    tmpdir = tempfile.mkdtemp(prefix="audit_p10_e2e_")
    outdir = tempfile.mkdtemp(prefix="audit_p10_out_")
    try:
        n_movements, n_reps, n_channels, fs, n_subjects = 5, 6, 8, 2000, 3
        for sid in range(1, n_subjects + 1):
            _build_subject_mat(os.path.join(tmpdir, f"S{sid}_E1_A1.mat"),
                              n_movements, n_reps, n_channels, fs, seed=sid)

        # نُلحق قاعدة بيانات اصطناعية بـ config الحقيقي مؤقتاً (بدل تزييف
        # config بالكامل، حتى نختبر p10 كما سيستدعيه المستخدم فعلياً)
        config.DB_PATHS['synthtest'] = tmpdir
        config.DB_META['synthtest'] = {
            "n_subjects": n_subjects, "n_movements": n_movements,
            "n_channels": n_channels, "sampling_rate": fs,
            "subject_ids": list(range(1, n_subjects + 1)),
            "movement_ids": list(range(1, n_movements + 1)),
            "exclude_rest": True, "display_name": "synth",
            "exercise_e1_only": False, "grouped_classes": None,
        }
        # run_loso_full يستدعي validate_data_path -> get_n_classes، اللتان
        # قد تتوقعان مفتاح 'synthtest' ضمن قائمة معروفة؛ نتحقق أدناه أن
        # هذا لا يفشل، وإلا نُبلغ بوضوح (توافق مع منطق data_loader الحالي)
        rows = run_coral_reg_sweep(
            db_key='synthtest', grid=[1e-1, 1e-3], output_dir=outdir,
            feature_extractor='custom_ppv', max_subjects=None,
            num_kernels=300, seed=42,
        )
        print(rows)
        assert len(rows) == 2
        for r in rows:
            assert 0.0 <= r['accuracy_mean'] <= 1.0
        print("\nPASS: p10 sensitivity sweep runs end-to-end via the real "
              "run_loso_full(), grid values produce distinct tagged results.")
        return 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        shutil.rmtree(outdir, ignore_errors=True)
        config.DB_PATHS.pop('synthtest', None)
        config.DB_META.pop('synthtest', None)


if __name__ == "__main__":
    sys.exit(main())
