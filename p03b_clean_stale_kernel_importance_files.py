"""
p03b_clean_stale_kernel_importance_files.py — تنظيف ملفات .npz التالفة (الطبقة 1 من الحل)
====================================================================
يفحص كل ملف kernel_imp_*.npz في FEATURES_DIR، ويتحقق أن:
    importance_matrix.shape[1] == len(dilations)

إذا لم يتطابق (كما يحدث دائماً مع TCA/SA لأنهما يُسقطان الأبعاد
إلى n_components)، يُحذف الملف التالف فوراً. هذا يمنع أي كود لاحق
(day4، أو أي سكريبت مستقبلي) من محاولة قراءة بيانات متعارضة الأبعاد.

هذا السكريبت idempotent وآمن للتشغيل في كل مرة — لا يؤثر على
الملفات السليمة (raw/centering/coral) إطلاقاً.
"""
import sys
import os
import glob
import numpy as np

sys.path.insert(0, '.')
import config


def clean_stale_npz(features_dir=None):
    features_dir = features_dir or config.FEATURES_DIR
    if not os.path.exists(features_dir):
        print(f"  [Clean] {features_dir} not found. Nothing to clean.", flush=True)
        return 0

    npz_files = sorted(glob.glob(os.path.join(features_dir, "kernel_imp_*.npz")))
    if not npz_files:
        print(f"  [Clean] No kernel_imp_*.npz files found in {features_dir}.", flush=True)
        return 0

    n_removed = 0
    n_ok = 0
    for path in npz_files:
        fname = os.path.basename(path)
        try:
            data = np.load(path)
            imp = data['importance_matrix']
            dil = data['dilations']
            if imp.shape[1] != len(dil):
                print(f"  [Clean] REMOVING stale/corrupted: {fname} "
                      f"(importance width={imp.shape[1]} != dilations len={len(dil)}) "
                      f"-- likely TCA/SA from before the dimensionality fix.",
                      flush=True)
                data.close()
                os.remove(path)
                n_removed += 1
            else:
                n_ok += 1
        except Exception as e:
            print(f"  [Clean] REMOVING unreadable/corrupted file: {fname} ({e})",
                  flush=True)
            try:
                os.remove(path)
                n_removed += 1
            except Exception:
                pass

    print(f"  [Clean] Scan complete: {n_ok} valid, {n_removed} removed "
          f"(out of {len(npz_files)} total).", flush=True)
    return n_removed


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--features_dir', type=str, default=None)
    args = p.parse_args()
    clean_stale_npz(args.features_dir)