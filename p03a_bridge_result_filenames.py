"""
p03a_bridge_result_filenames.py — جسر التوافق الكامل (أسماء ملفات + مفاتيح JSON)
============================================================================
يُصلح فجوتين بين مخرجات p01a_run_main_loso_benchmark.py ومتطلبات day3/5/6:

  FIX-1 (اسم الملف):
    عندنا:    results_db7_raw.json
    متوقَّع:  MiniROCKET_db7_raw_results.json

  FIX-2 (مفاتيح domain_shift):
    عندنا:    {'cov_before': ..., 'cov_after': ...}
    متوقَّع:  {'cov_frobenius_before': ..., 'cov_frobenius_after': ...}
    (p04b_compute_domain_shift_metrics.py يقرأ 'cov_frobenius_before/after' حصراً)

كلا الإصلاحين يتمّان بنسخ الملف (لا نلمس الأصل) + تعديل بنية JSON
في النسخة الجديدة فقط. آمن ويمكن إعادة تشغيله عدة مرات (idempotent).
"""
import os
import sys
import json
import shutil

sys.path.insert(0, '.')
import config


def _patch_domain_shift_keys(result):
    """يُعيد تسمية cov_before/cov_after -> cov_frobenius_before/after
    داخل كل عنصر من per_subject، إن وُجدت."""
    changed = False
    for subj in result.get('per_subject', []):
        ds = subj.get('domain_shift')
        if not isinstance(ds, dict):
            continue
        if 'cov_before' in ds and 'cov_frobenius_before' not in ds:
            ds['cov_frobenius_before'] = ds.pop('cov_before')
            changed = True
        if 'cov_after' in ds and 'cov_frobenius_after' not in ds:
            ds['cov_frobenius_after'] = ds.pop('cov_after')
            changed = True
    return result, changed


def bridge_filenames(results_dir=None):
    results_dir = results_dir or config.RESULTS_DIR
    if not os.path.exists(results_dir):
        print(f"  [WARN] {results_dir} not found. Skipping.")
        return 0

    n_bridged = 0
    n_key_fixed = 0

    for fname in sorted(os.listdir(results_dir)):
        if not fname.startswith("results_") or not fname.endswith(".json"):
            continue
        if fname == "MASTER_SUMMARY.json":
            continue

        core = fname[len("results_"):-len(".json")]
        target_name = f"MiniROCKET_{core}_results.json"
        src_path    = os.path.join(results_dir, fname)
        dst_path    = os.path.join(results_dir, target_name)

        # idempotent: أعد المعالجة فقط إذا الهدف غائب أو الأصل أحدث
        need_copy = True
        if os.path.exists(dst_path):
            if os.path.getmtime(dst_path) >= os.path.getmtime(src_path):
                need_copy = False

        if not need_copy:
            continue

        # حمّل، أصلح المفاتيح، احفظ بالاسم الجديد
        try:
            with open(src_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
        except Exception as e:
            print(f"  [WARN] Could not read {fname}: {e}")
            continue

        result, key_changed = _patch_domain_shift_keys(result)
        if key_changed:
            n_key_fixed += 1

        with open(dst_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str)

        n_bridged += 1

    print(f"  [Bridge] {n_bridged} file(s) renamed -> MiniROCKET_*_results.json")
    print(f"  [Bridge] {n_key_fixed} file(s) had domain_shift keys patched "
          f"(cov_before -> cov_frobenius_before)")
    print(f"  [Bridge] Directory: {results_dir}")
    return n_bridged


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--results_dir', type=str, default=None)
    args = p.parse_args()
    bridge_filenames(args.results_dir)
