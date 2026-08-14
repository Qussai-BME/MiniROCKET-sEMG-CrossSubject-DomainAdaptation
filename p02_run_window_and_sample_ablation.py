"""
p02_run_window_and_sample_ablation.py — Paper 2: Ablation Studies
=====================================================
يغطي دراستين ضروريتين لمستوى Q1:

  Study A — Window Size Ablation (DB7 + raw فقط)
    النوافذ: [100, 200, 400, 800] ms
    يجيب: ما حجم النافذة الأمثل لـ MiniROCKET مع EMG cross-subject؟
    الثابت: 50k samples, 10k kernels

  Study B — Sample Size Ablation (DB7 + raw فقط)
    الأحجام: [10_000, 25_000, 50_000] عينة
    يجيب: هل النتائج مستقرة عبر أحجام التدريب؟ (مطلوب لـ Q1)
    الثابت: 200ms, 10k kernels

كلا الدراستين تستخدم --resume تلقائياً.
المخرجات في config.ABLATION_DIR.
"""
import sys
import os
import json
import time
import gc
import warnings
from datetime import datetime

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import config
from p01a_run_main_loso_benchmark import run_loso_full

warnings.filterwarnings('ignore')

# ====================================================================
# إعدادات الـ Ablation
# ====================================================================

# Study A: Window sizes (ms)
WINDOW_SIZES   = [100, 200, 400, 800]

# Study B: Sample sizes
SAMPLE_SIZES   = [10_000, 25_000, 50_000]

# ثوابت مشتركة
DB_KEY         = 'db7'
METHOD         = 'raw'
NUM_KERNELS    = 10_000
OVERLAP        = 0.5
MAX_SAMPLES_A  = 50_000   # ثابت في Study A
WINDOW_MS_B    = 200      # ثابت في Study B
PER_SUBJ_LIMIT = 5_000


# ====================================================================
# Study A: Window Size Ablation
# ====================================================================
def run_window_ablation(output_dir=None, resume=True):
    out = output_dir or config.ABLATION_DIR
    os.makedirs(out, exist_ok=True)

    print('\n' + '#' * 72, flush=True)
    print('  STUDY A: Window Size Ablation', flush=True)
    print(f'  DB={DB_KEY}, method={METHOD}, samples={MAX_SAMPLES_A:,}', flush=True)
    print(f'  Windows: {WINDOW_SIZES} ms', flush=True)
    print('#' * 72, flush=True)

    results = {}
    for w_ms in WINDOW_SIZES:
        label = f"win{w_ms}ms"
        print(f"\n  >> Window = {w_ms} ms", flush=True)
        try:
            res = run_loso_full(
                db_key=DB_KEY,
                method=METHOD,
                num_kernels=NUM_KERNELS,
                output_dir=out,
                window_ms=w_ms,
                overlap=OVERLAP,
                resume=resume,
                max_train_samples=MAX_SAMPLES_A,
                per_subject_limit=PER_SUBJ_LIMIT,
                tag_suffix=label,
            )
            s = res['summary']
            results[label] = {
                'window_ms': w_ms,
                'mean':      s['mean'],
                'std':       s['std'],
                'median':    s['median'],
                'ci95_lower': s['ci95_lower'],
                'ci95_upper': s['ci95_upper'],
                'random_pct': s['avg_random_baseline_pct'],
                'ratio':      s['avg_ratio_vs_random'],
                'n_folds':    s['n'],
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ERROR (window={w_ms}ms): {e}", flush=True)
            results[label] = {'window_ms': w_ms, 'error': str(e)}

    # جدول ملخص
    print(f"\n{'='*72}", flush=True)
    print("  STUDY A RESULTS — Window Size Ablation", flush=True)
    print(f"  {'Window':>8} {'Acc':>8} {'±Std':>7} "
          f"{'Median':>8} {'Ratio':>7}", flush=True)
    print(f"  {'-'*45}", flush=True)
    for label, v in results.items():
        if 'error' in v:
            print(f"  {v['window_ms']:>5}ms  ERROR: {v['error']}", flush=True)
        else:
            print(f"  {v['window_ms']:>5}ms  "
                  f"{v['mean']:>7.4f}  {v['std']:>6.4f}  "
                  f"{v['median']:>7.4f}  {v['ratio']:>5.1f}x", flush=True)
    print('=' * 72, flush=True)

    # حفظ
    out_path = os.path.join(out, 'ablation_window_sizes.json')
    with open(out_path, 'w') as f:
        json.dump({
            'study': 'window_size_ablation',
            'db':    DB_KEY, 'method': METHOD,
            'num_kernels': NUM_KERNELS,
            'max_train_samples': MAX_SAMPLES_A,
            'timestamp': datetime.now().isoformat(),
            'results': results,
        }, f, indent=2)
    print(f"\n  Saved -> {out_path}", flush=True)
    return results


# ====================================================================
# Study B: Sample Size Ablation
# ====================================================================
def run_sample_ablation(output_dir=None, resume=True):
    out = output_dir or config.ABLATION_DIR
    os.makedirs(out, exist_ok=True)

    print('\n' + '#' * 72, flush=True)
    print('  STUDY B: Sample Size Ablation', flush=True)
    print(f'  DB={DB_KEY}, method={METHOD}, window={WINDOW_MS_B}ms', flush=True)
    print(f'  Samples: {SAMPLE_SIZES}', flush=True)
    print('#' * 72, flush=True)

    results = {}
    for n_samp in SAMPLE_SIZES:
        label = f"n{n_samp//1000}k"
        per_subj = min(PER_SUBJ_LIMIT, n_samp // 5)   # نسبة معقولة
        print(f"\n  >> Samples = {n_samp:,}", flush=True)
        try:
            res = run_loso_full(
                db_key=DB_KEY,
                method=METHOD,
                num_kernels=NUM_KERNELS,
                output_dir=out,
                window_ms=WINDOW_MS_B,
                overlap=OVERLAP,
                resume=resume,
                max_train_samples=n_samp,
                per_subject_limit=per_subj,
                tag_suffix=label,
            )
            s = res['summary']
            results[label] = {
                'n_samples': n_samp,
                'mean':      s['mean'],
                'std':       s['std'],
                'median':    s['median'],
                'ci95_lower': s['ci95_lower'],
                'ci95_upper': s['ci95_upper'],
                'random_pct': s['avg_random_baseline_pct'],
                'ratio':      s['avg_ratio_vs_random'],
                'n_folds':    s['n'],
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ERROR (n={n_samp}): {e}", flush=True)
            results[label] = {'n_samples': n_samp, 'error': str(e)}

    # جدول ملخص
    print(f"\n{'='*72}", flush=True)
    print("  STUDY B RESULTS — Sample Size Ablation", flush=True)
    print(f"  {'Samples':>10} {'Acc':>8} {'±Std':>7} "
          f"{'Median':>8} {'Ratio':>7}", flush=True)
    print(f"  {'-'*48}", flush=True)
    for label, v in results.items():
        if 'error' in v:
            print(f"  {v['n_samples']:>8,}  ERROR: {v['error']}", flush=True)
        else:
            print(f"  {v['n_samples']:>8,}  "
                  f"{v['mean']:>7.4f}  {v['std']:>6.4f}  "
                  f"{v['median']:>7.4f}  {v['ratio']:>5.1f}x", flush=True)
    print('=' * 72, flush=True)

    # حفظ
    out_path = os.path.join(out, 'ablation_sample_sizes.json')
    with open(out_path, 'w') as f:
        json.dump({
            'study': 'sample_size_ablation',
            'db':    DB_KEY, 'method': METHOD,
            'window_ms': WINDOW_MS_B,
            'num_kernels': NUM_KERNELS,
            'timestamp': datetime.now().isoformat(),
            'results': results,
        }, f, indent=2)
    print(f"\n  Saved -> {out_path}", flush=True)
    return results


# ====================================================================
# CLI
# ====================================================================
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(
        description='Paper 2: Ablation Studies (Window + Sample Size)'
    )
    p.add_argument('--study',
                   choices=['window', 'sample', 'both'],
                   default='both',
                   help='أي دراسة تشغّل (الافتراضي: كلتيهما)')
    p.add_argument('--output_dir', type=str, default=None)
    p.add_argument('--no_resume',  action='store_true',
                   help='إيقاف الـ resume (ابدأ من الصفر)')
    args = p.parse_args()

    resume = not args.no_resume
    out    = args.output_dir

    print('=' * 72, flush=True)
    print('  PAPER 2: Ablation Studies', flush=True)
    print(f'  Study={args.study}  Resume={resume}', flush=True)
    print('=' * 72, flush=True)

    if args.study in ('window', 'both'):
        run_window_ablation(out, resume=resume)

    if args.study in ('sample', 'both'):
        run_sample_ablation(out, resume=resume)

    print('\n  ALL ABLATION STUDIES COMPLETE', flush=True)