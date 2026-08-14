"""
run_pipeline_from_stage03.py — تشغيل تلقائي كامل (LIVE OUTPUT، لا تجميد ظاهري)
================================================================================
الإصلاح الجوهري عن النسخة السابقة:
  قبل: subprocess.run(stdout=PIPE) يُخزّن كل الخرج ولا يطبعه إلا بعد
       انتهاء العملية بالكامل -> يبدو للمستخدم أن السكريبت "توقف"
       رغم أنه يعمل بشكل طبيعي في الخلفية.
  بعد: subprocess.Popen + قراءة سطراً بسطر (streaming) -> كل طباعة
       من السكريبتات الفرعية تظهر فوراً في نفس اللحظة التي تُنتَج فيها.

الترتيب:
  1. p03a_bridge_result_filenames.py             (جسر التسمية + مفاتيح JSON)
  2. p03b_clean_stale_kernel_importance_files.py (تنظيف ملفات .npz تالفة)
  3. p04a_extract_kernel_importance_data.py       (نسخة مخففة: ~35-40 دقيقة)
  4. p04b_compute_domain_shift_metrics.py
  5. p05_analyze_kernel_importance.py
  6. p06_run_statistical_analysis.py              (حرج)
  7. p07_generate_all_figures.py
"""
import sys
import os
import subprocess
import argparse
import time
from datetime import datetime

sys.path.insert(0, '.')
import config

PYTHON = sys.executable

STEPS = [
    {
        'name': 'Bridge result filenames',
        'cmd':  [PYTHON, '-u', 'p03a_bridge_result_filenames.py'],
        'critical': True,
    },
    {
        'name': 'Clean stale/corrupted kernel importance files',
        'cmd':  [PYTHON, '-u', 'p03b_clean_stale_kernel_importance_files.py'],
        'critical': False,
    },
    {
        'name': 'Extract kernel importance data (.npz) [light: ~35-40min]',
        'cmd':  [PYTHON, '-u', 'p04a_extract_kernel_importance_data.py'],
        'critical': False,
    },
    {
        'name': 'Day 3: Domain shift metrics',
        'cmd':  [PYTHON, '-u', 'p04b_compute_domain_shift_metrics.py'],
        'critical': False,
    },
    {
        'name': 'Day 4: Kernel importance analysis',
        'cmd':  [PYTHON, '-u', 'p05_analyze_kernel_importance.py'],
        'critical': False,
    },
    {
        'name': 'Day 5: Statistical analysis',
        'cmd':  [PYTHON, '-u', 'p06_run_statistical_analysis.py', '--skip_ablation'],
        'critical': True,
    },
    {
        'name': 'Day 6: All visualizations',
        'cmd':  [PYTHON, '-u', 'p07_generate_all_figures.py', '--skip_ablation'],
        'critical': False,
    },
]


def run_step_live(step, log_f, dry_run=False):
    """يُشغّل subprocess ويطبع كل سطر فوراً (streaming) + يكتبه في اللوج."""
    name = step['name']
    cmd  = step['cmd']
    header = f"\n{'#'*72}\n  STEP: {name}\n  CMD:  {' '.join(cmd)}\n{'#'*72}"
    print(header, flush=True)
    log_f.write(header + "\n")
    log_f.flush()

    if dry_run:
        print("  [DRY RUN] Skipped actual execution.", flush=True)
        return True

    t0 = time.time()
    try:
        # حماية جذرية: نفرض UTF-8 على كل عملية فرعية بغض النظر عن
        # ترميز طرفية Windows (cp1256 وغيره لا يدعمان كل رموز اليونيكود).
        # هذا يمنع أي UnicodeEncodeError مستقبلي حتى لو تسلل رمز
        # يونيكود جديد لأي ملف لم نفحصه.
        child_env = os.environ.copy()
        child_env['PYTHONIOENCODING'] = 'utf-8'
        child_env['PYTHONUTF8'] = '1'

        # -u = unbuffered على مستوى بايثون الفرعي (يمنع تأخر الطباعة)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,                 # line-buffered
            env=child_env,
        )

        # قراءة حية سطراً بسطر — تظهر فوراً في الطرفية وتُكتب في اللوج
        for line in iter(proc.stdout.readline, ''):
            print(line, end='', flush=True)
            log_f.write(line)
            log_f.flush()

        proc.stdout.close()
        returncode = proc.wait()
        elapsed = time.time() - t0

        if returncode != 0:
            msg = f"\n  [FAILED] {name} exited with code {returncode} ({elapsed:.1f}s)"
            print(msg, flush=True)
            log_f.write(msg + "\n")
            return False

        msg = f"\n  [OK] {name} completed in {elapsed:.1f}s ({elapsed/60:.1f} min)"
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()
        return True

    except Exception as e:
        msg = f"\n  [EXCEPTION] {name}: {e}"
        print(msg, flush=True)
        log_f.write(msg + "\n")
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--skip_tsne', action='store_true')
    p.add_argument('--dry_run',   action='store_true')
    args = p.parse_args()

    for step in STEPS:
        if 'p07_generate_all_figures.py' in step['cmd'] and args.skip_tsne:
            step['cmd'].append('--skip_tsne')

    os.makedirs(config.LOG_DIR, exist_ok=True)
    log_path = os.path.join(
        config.LOG_DIR,
        f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    banner = (
        f"{'='*72}\n"
        f"  PAPER 2: Remaining Pipeline (Day 3 -> Day 6) — LIVE OUTPUT\n"
        f"  Log: {log_path}\n"
        f"  Dry run: {args.dry_run}\n"
        f"  NOTE: Output now streams live, line by line, as it happens.\n"
        f"        If a step goes quiet for a long time, that step is\n"
        f"        genuinely still computing (e.g. MiniROCKET transform)\n"
        f"        — not frozen. Progress prints appear per fold.\n"
        f"{'='*72}"
    )
    print(banner, flush=True)

    t_start = time.time()
    results = {}

    with open(log_path, 'w', encoding='utf-8') as log_f:
        log_f.write(banner + "\n")
        for step in STEPS:
            ok = run_step_live(step, log_f, dry_run=args.dry_run)
            results[step['name']] = ok

            if not ok and step['critical']:
                stop_msg = (
                    f"\n{'!'*72}\n"
                    f"  STOPPED: Critical step failed: {step['name']}\n"
                    f"  Fix the issue, then resume with the SAME command.\n"
                    f"  Full log: {log_path}\n"
                    f"{'!'*72}"
                )
                print(stop_msg, flush=True)
                log_f.write(stop_msg + "\n")
                sys.exit(1)
            elif not ok:
                warn_msg = (f"\n  [WARN] Non-critical step failed: {step['name']}. "
                           f"Continuing with remaining steps.")
                print(warn_msg, flush=True)
                log_f.write(warn_msg + "\n")

    elapsed_total = time.time() - t_start
    summary = [f"\n{'='*72}", "  PIPELINE SUMMARY", f"{'='*72}"]
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        summary.append(f"  [{status:>6}] {name}")
    summary.append(f"\n  Total time: {elapsed_total/60:.1f} minutes")
    summary.append(f"  Full log:   {log_path}")
    summary.append(f"  Outputs in: {config.OUTPUT_DIR}")
    summary.append('=' * 72)
    final_text = "\n".join(summary)
    print(final_text, flush=True)


if __name__ == '__main__':
    main()