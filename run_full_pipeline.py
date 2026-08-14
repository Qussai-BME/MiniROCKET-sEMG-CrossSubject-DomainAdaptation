"""
run_full_pipeline.py — Paper 2 V2: GOLDEN PIPELINE ORCHESTRATOR
============================================================
هذا هو الملف الرئيسي لتشغيل المشروع بالكامل بالتسلسل الصحيح.

يقوم هذا المنسق بتشغيل الأيام التالية:
  - Day 2:  p01a_run_main_loso_benchmark.py      (15 تجربة LOSO رئيسية)
  - Day 2b: p02_run_window_and_sample_ablation.py      (11 تجربة إزالة)
  - Day 3:  p04b_compute_domain_shift_metrics.py  (جداول انحراف المجال)
  - Day 4:  p05_analyze_kernel_importance.py     (تحليل أهمية النوى)
  - Day 5:  p06_run_statistical_analysis.py            (التحليل الإحصائي النهائي)
  - Day 6:  p07_generate_all_figures.py        (جميع الأشكال البيانية)

الميزات:
  1. تشغيل الكل بالتسلسل.
  2. تشغيل يوم محدد (--day).
  3. البدء من يوم معين (--from-day).
  4. تمرير المعاملات إلى الأيام الفرعية (--all, --resume, --skip-*).
  5. خيار التجربة الجافة (--dry-run) لمعاينة الأوامر.
  6. تحقق من وجود الأكواد قبل التشغيل.
  7. إيقاف التشغيل عند أول خطأ مع رسالة استئناف واضحة.

الاستخدام:
  # تشغيل كل شيء
  python run_full_pipeline.py

  # تشغيل يوم 5 فقط
  python run_full_pipeline.py --day 5

  # البدء من يوم 3 (تخطي يوم 2 ويوم 2ب)
  python run_full_pipeline.py --from-day 3

  # تشغيل كل شيء مع استئناف يوم 2 وتخطي t-SNE
  python run_full_pipeline.py --resume --skip-tsne

  # معاينة الأوامر دون تشغيل فعلي
  python run_full_pipeline.py --dry-run
"""
import sys
import os
import subprocess
import argparse
import time
from datetime import datetime

# ====================================================================
# 1. إعدادات المسار الأساسية
# ====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# قائمة الأيام مع معلوماتها
DAYS = {
    '2': {
        'script': 'p01a_run_main_loso_benchmark.py',
        'description': 'Day 2: Main LOSO Experiments (15 experiments)',
        'default_args': ['--all'],
        'extra_args': ['--resume'],
        'depends_on': None,
    },
    '2b': {
        'script': 'p02_run_window_and_sample_ablation.py',
        'description': 'Day 2b: Ablation Studies (11 experiments)',
        'default_args': [],
        'extra_args': [],
        'depends_on': '2',
    },
    '3': {
        'script': 'p04b_compute_domain_shift_metrics.py',
        'description': 'Day 3: Domain Shift Metrics (CSVs + figure)',
        'default_args': [],
        'extra_args': [],
        'depends_on': '2',
    },
    '4': {
        'script': 'p05_analyze_kernel_importance.py',
        'description': 'Day 4: Kernel Importance Analysis (CSVs + figures)',
        'default_args': [],
        'extra_args': ['--skip_plot'],
        'depends_on': '2',
    },
    '5': {
        'script': 'p06_run_statistical_analysis.py',
        'description': 'Day 5: Statistical Analysis (8 tables + summary)',
        'default_args': [],
        'extra_args': ['--skip_ablation'],
        'depends_on': ['2', '2b'],
    },
    '6': {
        'script': 'p07_generate_all_figures.py',
        'description': 'Day 6: All Figures (12+ publication-quality figures)',
        'default_args': [],
        'extra_args': ['--skip_tsne', '--skip_ablation'],
        'depends_on': ['2', '2b', '3', '4', '5'],
    },
}

# ترتيب التشغيل
ORDER = ['2', '2b', '3', '4', '5', '6']

# ====================================================================
# 2. دوال التحقق والتنفيذ
# ====================================================================
def check_script_exists(script_name):
    """
    التحقق من وجود السكربت في المجلد الحالي.

    المعاملات:
        script_name (str): اسم السكربت.

    المخرجات:
        bool: True إذا كان الملف موجوداً.
    """
    script_path = os.path.join(SCRIPT_DIR, script_name)
    return os.path.exists(script_path)

def run_script(script_name, args_list, dry_run=False):
    """
    تشغيل سكربت معين باستخدام subprocess.

    المعاملات:
        script_name (str): اسم السكربت.
        args_list (list): قائمة المعاملات.
        dry_run (bool): إذا كان True، يطبع الأمر فقط دون تنفيذ.

    المخرجات:
        bool: True إذا نجح التشغيل، False إذا فشل.
    """
    script_path = os.path.join(SCRIPT_DIR, script_name)

    if not os.path.exists(script_path):
        print(f"  [ERROR] Script not found: {script_path}", flush=True)
        return False

    cmd = [sys.executable, script_path] + args_list
    cmd_str = ' '.join(cmd)

    print(f"\n  {'[DRY RUN] ' if dry_run else ''}Executing: {cmd_str}", flush=True)

    if dry_run:
        return True

    try:
        result = subprocess.run(cmd, cwd=SCRIPT_DIR)
        if result.returncode != 0:
            print(f"  [ERROR] {script_name} failed with exit code {result.returncode}", flush=True)
            return False
        return True
    except Exception as e:
        print(f"  [ERROR] Exception while running {script_name}: {e}", flush=True)
        return False

def check_dependencies(day_key, args):
    """
    التحقق من وجود السكربتات المعتمدة على اليوم.

    المعاملات:
        day_key (str): مفتاح اليوم.
        args (Namespace): معاملات سطر الأوامر.

    المخرجات:
        bool: True إذا كانت جميع التبعيات موجودة.
    """
    day_info = DAYS.get(day_key)
    if day_info is None:
        return False

    depends = day_info.get('depends_on')
    if depends is None:
        return True

    # تحويل إلى قائمة إذا كانت مفردة
    if isinstance(depends, str):
        depends = [depends]

    for dep in depends:
        # التحقق من وجود السكربت المعتمد عليه
        dep_script = DAYS.get(dep, {}).get('script')
        if dep_script and not check_script_exists(dep_script):
            print(f"  [WARN] Dependency script not found: {dep_script}", flush=True)
            # نستمر مع تحذير بدلاً من فشل كامل

    return True

# ====================================================================
# 3. بناء معاملات كل يوم
# ====================================================================
def build_day_args(day_key, args):
    """
    بناء قائمة المعاملات ليوم معين بناءً على معاملات المستخدم.

    المعاملات:
        day_key (str): مفتاح اليوم.
        args (Namespace): معاملات سطر الأوامر.

    المخرجات:
        list: قائمة المعاملات للسكربت.
    """
    day_info = DAYS.get(day_key)
    if day_info is None:
        return []

    # المعاملات الافتراضية
    cmd_args = list(day_info.get('default_args', []))

    # إضافة معاملات إضافية خاصة باليوم
    if day_key == '2':
        # اليوم 2: إضافة --resume إذا طلب المستخدم
        if args.resume:
            cmd_args.append('--resume')
        # إضافة exercise_filter إذا تم تحديده
        if args.exercise_filter:
            cmd_args.extend(['--exercise_filter', args.exercise_filter])
        # إضافة group_classes
        if args.group_classes:
            cmd_args.append('--group_classes')
        # إضافة num_kernels
        if args.num_kernels != 10000:
            cmd_args.extend(['--num_kernels', str(args.num_kernels)])
        # إضافة window_ms
        if args.window_ms != 200:
            cmd_args.extend(['--window_ms', str(args.window_ms)])

    elif day_key == '2b':
        # اليوم 2ب: إضافة خيارات التخطي
        if args.skip_window:
            cmd_args.append('--skip-window')
        if args.skip_kernels:
            cmd_args.append('--skip-kernels')
        if args.skip_coral_reg:
            cmd_args.append('--skip-coral-reg')
        if args.skip_grouping:
            cmd_args.append('--skip-grouping')
        if args.resume:
            cmd_args.append('--resume')
        # تمرير قاعدة البيانات
        if args.db:
            for db in args.db:
                cmd_args.extend(['--db', db])

    elif day_key == '3':
        # اليوم 3: لا توجد معاملات خاصة
        pass

    elif day_key == '4':
        # اليوم 4: إضافة skip_plot
        if args.skip_plot_kernel:
            cmd_args.append('--skip_plot')

    elif day_key == '5':
        # اليوم 5: إضافة skip_ablation
        if args.skip_ablation_stats:
            cmd_args.append('--skip_ablation')

    elif day_key == '6':
        # اليوم 6: إضافة خيارات التخطي
        if args.skip_tsne:
            cmd_args.append('--skip_tsne')
        if args.skip_ablation_fig:
            cmd_args.append('--skip_ablation')
        if args.skip_plot_kernel:
            cmd_args.append('--skip_plot')

    # تمرير مجلد النتائج إذا تم تحديده
    if args.results_dir:
        cmd_args.extend(['--results_dir', args.results_dir])

    # تمرير مجلد المخرجات إذا تم تحديده
    if args.output_dir:
        cmd_args.extend(['--output_dir', args.output_dir])

    # تمرير مجلد الأشكال إذا تم تحديده
    if args.figures_dir:
        cmd_args.extend(['--figures_dir', args.figures_dir])

    return cmd_args

# ====================================================================
# 4. واجهة سطر الأوامر (CLI)
# ====================================================================
def parse_args():
    """
    تحليل معاملات سطر الأوامر للمنسق الرئيسي.
    """
    parser = argparse.ArgumentParser(
        description="Paper 2 V2: Golden Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
أمثلة:
  # تشغيل كل شيء
  python run_full_pipeline.py

  # تشغيل يوم 5 فقط
  python run_full_pipeline.py --day 5

  # البدء من يوم 3 (تخطي يوم 2 ويوم 2ب)
  python run_full_pipeline.py --from-day 3

  # تشغيل كل شيء مع استئناف يوم 2 وتخطي t-SNE
  python run_full_pipeline.py --resume --skip-tsne

  # معاينة الأوامر دون تشغيل فعلي
  python run_full_pipeline.py --dry-run

الأيام المتاحة:
  2  - Main LOSO Experiments (15 تجربة)
  2b - Ablation Studies (11 تجربة)
  3  - Domain Shift Metrics
  4  - Kernel Importance Analysis
  5  - Statistical Analysis
  6  - All Figures
        """
    )

    # ── معاملات التشغيل العامة ──
    parser.add_argument("--day", type=str, default=None,
                        choices=list(DAYS.keys()),
                        help="تشغيل يوم محدد فقط")

    parser.add_argument("--from-day", type=str, default=None,
                        choices=list(DAYS.keys()),
                        help="البدء من يوم معين")

    parser.add_argument("--dry-run", action="store_true",
                        help="عرض الأوامر فقط دون تنفيذ")

    parser.add_argument("--resume", action="store_true",
                        help="تمرير --resume إلى الأيام التي تدعمه (2, 2b)")

    # ── معاملات اليوم 2 ──
    parser.add_argument("--exercise_filter", type=str, default=None,
                        choices=["E1", "E2", "E1+E2", "all"],
                        help="تصفية التمرينات لليوم 2")

    parser.add_argument("--group_classes", action="store_true",
                        help="تجميع الأصناف لليوم 2")

    parser.add_argument("--num_kernels", type=int, default=10000,
                        help="عدد النوى (الافتراضي 10000)")

    parser.add_argument("--window_ms", type=int, default=200,
                        help="حجم النافذة بالميلي ثانية (الافتراضي 200)")

    parser.add_argument("--db", type=str, default=None,
                        choices=["db7", "db3", "db2"],
                        action='append',
                        help="قاعدة البيانات لليوم 2ب (يمكن تكرارها)")

    # ── معاملات اليوم 2ب ──
    parser.add_argument("--skip-window", action="store_true",
                        help="تخطي إزالة حجم النافذة")

    parser.add_argument("--skip-kernels", action="store_true",
                        help="تخطي إزالة عدد النوى")

    parser.add_argument("--skip-coral-reg", action="store_true",
                        help="تخطي إزالة معامل تنظيم CORAL")

    parser.add_argument("--skip-grouping", action="store_true",
                        help="تخطي إزالة تجميع الأصناف")

    # ── معاملات اليوم 4 ──
    parser.add_argument("--skip-plot-kernel", action="store_true",
                        help="تخطي رسوم أهمية النوى")

    # ── معاملات اليوم 5 ──
    parser.add_argument("--skip-ablation-stats", action="store_true",
                        help="تخطي الإحصاء لتجارب الإزالة")

    # ── معاملات اليوم 6 ──
    parser.add_argument("--skip-tsne", action="store_true",
                        help="تخطي رسوم t-SNE (لتوفير الوقت)")

    parser.add_argument("--skip-ablation-fig", action="store_true",
                        help="تخطي رسوم الإزالة")

    # ── مجلدات المخرجات ──
    parser.add_argument("--results_dir", type=str, default=None,
                        help="مجلد النتائج")

    parser.add_argument("--output_dir", type=str, default=None,
                        help="مجلد المخرجات العامة")

    parser.add_argument("--figures_dir", type=str, default=None,
                        help="مجلد الأشكال")

    return parser.parse_args()

# ====================================================================
# 5. الدالة الرئيسية (Main)
# ====================================================================
def main():
    """
    النقطة الرئيسية لتشغيل المنسق.
    """
    args = parse_args()

    # ── طباعة معلومات البداية ──
    print("=" * 80, flush=True)
    print("  PAPER 2 V2: GOLDEN PIPELINE ORCHESTRATOR", flush=True)
    print("  Automated Pipeline for All Experiments and Analyses", flush=True)
    print("=" * 80, flush=True)
    print(f"  Timestamp:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"  Dry Run:        {args.dry_run}", flush=True)
    print(f"  Resume:         {args.resume}", flush=True)
    print(f"  Day:            {args.day or 'ALL'}", flush=True)
    print(f"  From Day:       {args.from_day or 'START'}", flush=True)
    print("=" * 80 + "\n", flush=True)

    # ── التحقق من وجود السكربتات الأساسية ──
    print("  Checking scripts...", flush=True)
    missing_scripts = []
    for day_key, day_info in DAYS.items():
        script_name = day_info['script']
        if not check_script_exists(script_name):
            missing_scripts.append(script_name)
            print(f"    [WARN] {script_name} not found", flush=True)

    if missing_scripts:
        print(f"\n  [WARNING] {len(missing_scripts)} scripts not found.", flush=True)
        print("  The pipeline may fail. Continuing anyway...", flush=True)

    # ── تحديد الأيام المراد تشغيلها ──
    if args.day:
        # تشغيل يوم محدد
        days_to_run = [args.day]
        print(f"\n  Running single day: {args.day}", flush=True)
    elif args.from_day:
        # البدء من يوم معين
        if args.from_day not in ORDER:
            print(f"  [ERROR] Unknown day: {args.from_day}", flush=True)
            return
        start_idx = ORDER.index(args.from_day)
        days_to_run = ORDER[start_idx:]
        print(f"\n  Running from day {args.from_day}: {days_to_run}", flush=True)
    else:
        # تشغيل كل شيء
        days_to_run = ORDER
        print("\n  Running full pipeline: ALL days", flush=True)

    print(f"  Days to run: {days_to_run}", flush=True)
    print("")

    # ── حلقة التشغيل ──
    total_start = time.time()
    total_days = len(days_to_run)
    completed_days = 0
    failed_day = None

    for idx, day_key in enumerate(days_to_run, 1):
        day_info = DAYS.get(day_key)
        if day_info is None:
            print(f"  [ERROR] Unknown day: {day_key}", flush=True)
            continue

        script_name = day_info['script']
        description = day_info['description']

        print(f"\n{'#' * 80}", flush=True)
        print(f"  [{idx}/{total_days}] {description}", flush=True)
        print(f"  Script: {script_name}", flush=True)
        print(f"{'#' * 80}\n", flush=True)

        # التحقق من التبعيات
        if not check_dependencies(day_key, args):
            print(f"  [WARN] Dependencies check failed for {day_key}", flush=True)

        # بناء المعاملات
        cmd_args = build_day_args(day_key, args)

        # تشغيل السكربت
        success = run_script(script_name, cmd_args, args.dry_run)

        if success:
            completed_days += 1
            print(f"\n  ✅ {description} completed successfully!", flush=True)
        else:
            failed_day = day_key
            print(f"\n  ❌ {description} FAILED!", flush=True)
            break

    # ── الملخص النهائي ──
    total_elapsed = time.time() - total_start
    hours = int(total_elapsed // 3600)
    minutes = int((total_elapsed % 3600) // 60)
    seconds = int(total_elapsed % 60)

    print("\n" + "=" * 80, flush=True)
    if failed_day is None:
        print("  🎉 PIPELINE COMPLETED SUCCESSFULLY!", flush=True)
        print(f"  Total days run: {completed_days}/{total_days}", flush=True)
    else:
        print("  ⚠️ PIPELINE STOPPED EARLY!", flush=True)
        print(f"  Failed on day: {failed_day}", flush=True)
        print(f"  Completed: {completed_days}/{total_days}", flush=True)
        print(f"\n  To resume from the failed day:", flush=True)
        print(f"    python run_full_pipeline.py --from-day {failed_day} --resume", flush=True)
        if failed_day == '2':
            print(f"    (or run the specific script: python {DAYS[failed_day]['script']} --all --resume)", flush=True)

    print(f"  Total time: {hours}h {minutes}m {seconds}s", flush=True)
    print("=" * 80 + "\n", flush=True)

# ====================================================================
# 6. نقطة الدخول
# ====================================================================
if __name__ == "__main__":
    main()