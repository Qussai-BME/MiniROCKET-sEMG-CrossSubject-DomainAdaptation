"""
audit_p08_e2e_smoke_test.py — verifies p08 multi-seed sweep runs end-to-end
"""
import os, sys, tempfile, shutil
import numpy as np
import scipy.io
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


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
    tmpdir = tempfile.mkdtemp(prefix="audit_p08_e2e_")
    outdir = tempfile.mkdtemp(prefix="audit_p08_out_")
    try:
        n_movements, n_reps, n_channels, fs, n_subjects = 5, 6, 8, 2000, 3
        for sid in range(1, n_subjects + 1):
            _build_subject_mat(os.path.join(tmpdir, f"S{sid}_E1_A1.mat"),
                              n_movements, n_reps, n_channels, fs, seed=sid)

        config.DB_PATHS['synthtest'] = tmpdir
        config.DB_META['synthtest'] = {
            "n_subjects": n_subjects, "n_movements": n_movements,
            "n_channels": n_channels, "sampling_rate": fs,
            "subject_ids": list(range(1, n_subjects + 1)),
            "movement_ids": list(range(1, n_movements + 1)),
            "exclude_rest": True, "display_name": "synth",
            "exercise_e1_only": False, "grouped_classes": None,
        }

        import argparse
        fake_args = argparse.Namespace(
            db="synthtest", method="raw", all=False,
            extractor="custom_ppv", seeds="1,2",
            num_kernels=300, max_train_samples=50_000,
            per_subject_limit=5_000, max_subjects=None,
            output_dir=outdir,
        )
        import importlib
        import p08_run_multiseed_sweep as p08
        importlib.reload(p08)
        with mock.patch.object(p08, "parse_args", return_value=fake_args):
            p08.main()

        summary_path = os.path.join(outdir, "MULTISEED_SUMMARY_1_2.json")
        assert os.path.exists(summary_path), "multiseed summary file missing"
        import json
        with open(summary_path) as f:
            data = json.load(f)
        assert data['seeds'] == [1, 2]
        key = list(data['results'].keys())[0]
        assert len(data['results'][key]['accuracy_per_seed']) == 2
        print("\nPASS: p08 multi-seed sweep runs end-to-end, produces "
              "per-seed results tagged distinctly, and aggregates them.")
        return 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        shutil.rmtree(outdir, ignore_errors=True)
        config.DB_PATHS.pop('synthtest', None)
        config.DB_META.pop('synthtest', None)


if __name__ == "__main__":
    sys.exit(main())
