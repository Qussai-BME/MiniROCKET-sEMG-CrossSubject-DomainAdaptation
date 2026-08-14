"""
p05_analyze_kernel_importance.py — Paper 2 V2: KERNEL IMPORTANCE ANALYSIS
===================================================================
هذا السكربت يقوم بتحليل أهمية نوى MiniROCKET باستخدام أسلوب Detach-ROCKET.

المبدأ:
  - بعد تدريب مصنف Ridge، يتم أخذ القيم المطلقة لمعاملات الانحدار (|coef|)
    كدليل على أهمية كل خاصية (PPV feature).
  - بما أن كل نواة تنتج عدداً من الخصائص يساوي عدد القنوات،
    يتم حساب متوسط الأهمية عبر القنوات للحصول على أهمية واحدة لكل نواة.
  - يتم تجميع النوى حسب قيمة التمدد (dilation) لمعرفة أي التمددات
    تحمل أكبر وزن تمييزي.

المخرجات:
  1. kernel_importance_{db}_{method}.csv — أهمية كل نواة (kernel_id, importance_mean, importance_std, dilation).
  2. kernel_importance_summary_{db}_{method}.csv — ملخص حسب التمدد (dilation, mean_importance, std_importance, n_kernels).
  3. kernel_importance_before_after_coral.csv — مقارنة raw vs coral (لكل قاعدة).
  4. أشكال بيانية: figure_kernel_importance_{db}_{method}.png — أهمية النوى حسب التمدد.

الاستخدام:
  python p05_analyze_kernel_importance.py
  python p05_analyze_kernel_importance.py --skip_plot
  python p05_analyze_kernel_importance.py --features_dir ./path/to/features
"""
import sys
import os
import numpy as np
import csv as _csv
import argparse
import warnings

# ====================================================================
# 1. إعدادات المسار والاستيرادات
# ====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import config

# تجاهل التحذيرات
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ====================================================================
# 2. دوال تحميل بيانات الأهمية
# ====================================================================
def load_kernel_importance(features_dir, tag):
    """
    تحميل مصفوفة أهمية النوى من ملف .npz.

    المعاملات:
        features_dir (str): مجلد الخصائص.
        tag (str): العلامة المميزة (مثل 'db7_raw').

    المخرجات:
        tuple: (importance_matrix, dilations) حيث importance_matrix هي
               مصفوفة (n_folds, n_kernels) و dilations هي (n_kernels,).
               في حال عدم وجود الملف، تُرجع (None, None).
    """
    npz_path = os.path.join(features_dir, f"kernel_imp_{tag}.npz")
    if not os.path.exists(npz_path):
        return None, None

    try:
        data = np.load(npz_path, allow_pickle=True)
        importance_matrix = data["importance_matrix"]  # (n_folds, n_kernels)
        dilations = data["dilations"]                  # (n_kernels,)

        # حماية دفاعية (الطبقة 3): تحقق من تطابق الأبعاد بغض النظر عن
        # مصدر الملف. يحدث هذا التعارض بشكل متوقع مع TCA/SA لأنهما
        # يُسقطان البيانات على n_components (مثلاً 50) وليس على فضاء
        # النوى الأصلي (مثلاً 3000) — تحليل "أهمية النواة حسب التمدد"
        # لا معنى له علمياً في هذه الحالة، لذا نتخطى الملف بأمان بدل
        # إسقاط السكريبت بالكامل عبر IndexError.
        if importance_matrix.shape[1] != len(dilations):
            print(f"  [SKIP] {tag}: dimension mismatch "
                  f"(importance width={importance_matrix.shape[1]} != "
                  f"dilations length={len(dilations)}). This typically "
                  f"means the method projects to a reduced feature space "
                  f"(e.g. TCA/SA n_components), so per-kernel dilation "
                  f"importance does not apply here.", flush=True)
            return None, None

        return importance_matrix, dilations
    except Exception as e:
        print(f"  [WARN] Could not load {npz_path}: {e}", flush=True)
        return None, None

# ====================================================================
# 3. بناء جداول CSV
# ====================================================================
def build_kernel_csv(db_key, method, imp_matrix, dilations, output_dir):
    """
    بناء جدول CSV لأهمية كل نواة على حدة.

    المعاملات:
        db_key (str): مفتاح قاعدة البيانات.
        method (str): طريقة التكيف.
        imp_matrix (np.ndarray): مصفوفة الأهمية (n_folds, n_kernels).
        dilations (np.ndarray): قيم التمدد (n_kernels,).
        output_dir (str): مجلد الإخراج.

    المخرجات:
        str: مسار الملف المُنشأ.
    """
    # حساب متوسط الأهمية والانحراف المعياري عبر الـ folds
    mean_imp = imp_matrix.mean(axis=0)
    std_imp = imp_matrix.std(axis=0)

    rows = []
    for i in range(len(dilations)):
        rows.append({
            "kernel_id": int(i),
            "importance_mean": round(float(mean_imp[i]), 6),
            "importance_std": round(float(std_imp[i]), 6),
            "dilation": int(dilations[i]),
        })

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"kernel_importance_{db_key}_{method}.csv")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=["kernel_id", "importance_mean", "importance_std", "dilation"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  [SAVED] {path}", flush=True)
    return path

def build_dilation_summary_csv(db_key, method, imp_matrix, dilations, output_dir):
    """
    بناء جدول CSV ملخص حسب قيمة التمدد (dilation).

    المعاملات:
        db_key (str): مفتاح قاعدة البيانات.
        method (str): طريقة التكيف.
        imp_matrix (np.ndarray): مصفوفة الأهمية (n_folds, n_kernels).
        dilations (np.ndarray): قيم التمدد (n_kernels,).
        output_dir (str): مجلد الإخراج.

    المخرجات:
        str: مسار الملف المُنشأ.
    """
    unique_dils = np.unique(dilations)
    rows = []

    for d in unique_dils:
        mask = (dilations == d)
        imp_d = imp_matrix[:, mask]  # (n_folds, n_kernels_with_d)
        rows.append({
            "dilation": int(d),
            "mean_importance": round(float(imp_d.mean()), 6),
            "std_importance": round(float(imp_d.std()), 6),
            "sum_importance": round(float(imp_d.sum()), 4),
            "n_kernels": int(mask.sum()),
        })

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"kernel_importance_summary_{db_key}_{method}.csv")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=["dilation", "mean_importance", "std_importance", "sum_importance", "n_kernels"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  [SAVED] {path}", flush=True)
    return path

def build_before_after_coral_csv(features_dir, db_key, output_dir):
    """
    بناء جدول مقارنة raw vs coral لأهمية النوى.

    المعاملات:
        features_dir (str): مجلد الخصائص.
        db_key (str): مفتاح قاعدة البيانات.
        output_dir (str): مجلد الإخراج.

    المخرجات:
        str: مسار الملف المُنشأ.
    """
    imp_raw, dil_raw = load_kernel_importance(features_dir, f"{db_key}_raw")
    imp_cor, dil_cor = load_kernel_importance(features_dir, f"{db_key}_coral")

    if imp_raw is None or imp_cor is None:
        print(f"  [SKIP] Before/after coral for {db_key} (data missing)", flush=True)
        return None

    # التأكد من أن عدد النوى متطابق
    if imp_raw.shape[1] != imp_cor.shape[1]:
        print(f"  [WARN] Kernel count mismatch for {db_key}: raw={imp_raw.shape[1]}, coral={imp_cor.shape[1]}", flush=True)
        return None

    mean_raw = imp_raw.mean(axis=0)
    mean_cor = imp_cor.mean(axis=0)

    rows = []
    for i in range(len(dil_raw)):
        change = mean_cor[i] - mean_raw[i]
        change_pct = (change / (abs(mean_raw[i]) + 1e-12)) * 100
        rows.append({
            "kernel_id": int(i),
            "importance_raw": round(float(mean_raw[i]), 6),
            "importance_coral": round(float(mean_cor[i]), 6),
            "change": round(float(change), 6),
            "change_pct": round(float(change_pct), 2),
            "dilation": int(dil_raw[i]),
        })

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"kernel_importance_before_after_coral_{db_key}.csv")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=["kernel_id", "importance_raw", "importance_coral",
                                                "change", "change_pct", "dilation"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  [SAVED] {path}", flush=True)
    return path

# ====================================================================
# 4. رسم الأشكال البيانية
# ====================================================================
def plot_kernel_importance_by_dilation(features_dir, db_key, method, output_dir):
    """
    رسم شكل بياني لأهمية النوى حسب قيمة التمدد.

    المعاملات:
        features_dir (str): مجلد الخصائص.
        db_key (str): مفتاح قاعدة البيانات.
        method (str): طريقة التكيف.
        output_dir (str): مجلد الإخراج.

    المخرجات:
        str: مسار الملف المُنشأ.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not available. Skipping figure generation.", flush=True)
        return None

    imp, dils = load_kernel_importance(features_dir, f"{db_key}_{method}")
    if imp is None:
        return None

    unique_d = np.sort(np.unique(dils))
    means = [float(imp[:, dils == d].mean()) for d in unique_d]
    stds = [float(imp[:, dils == d].std()) for d in unique_d]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=config.FIGURE_DPI)
    color = config.METHOD_COLORS.get(method, "#555555")

    bars = ax.bar(range(len(unique_d)), means, yerr=stds,
                  color=color, alpha=0.85, capsize=3,
                  error_kw={'elinewidth': 1, 'capthick': 1})

    ax.set_xticks(range(len(unique_d)))
    ax.set_xticklabels([str(int(d)) for d in unique_d], fontsize=9)
    ax.set_xlabel("Dilation Value", fontsize=12)
    ax.set_ylabel("Mean |Ridge Coefficient| (Importance)", fontsize=12)
    ax.set_title(f"Kernel Importance by Dilation\n"
                 f"{config.DB_META.get(db_key, {}).get('display_name', db_key)} -- "
                 f"{config.METHOD_LABELS.get(method, method)}",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"figure_kernel_importance_{db_key}_{method}.png")
    fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight", facecolor='white')
    plt.close(fig)
    print(f"  [SAVED] {save_path}", flush=True)

    return save_path

def plot_combined_kernel_importance(features_dir, db_key, output_dir):
    """
    رسم شكل بياني يجمع raw و coral في رسم واحد لقاعدة معينة.

    المعاملات:
        features_dir (str): مجلد الخصائص.
        db_key (str): مفتاح قاعدة البيانات.
        output_dir (str): مجلد الإخراج.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=config.FIGURE_DPI)

    for ax_idx, method in enumerate(["raw", "coral"]):
        ax = axes[ax_idx]
        imp, dils = load_kernel_importance(features_dir, f"{db_key}_{method}")
        if imp is None:
            ax.text(0.5, 0.5, f"No data for {method}", ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(config.METHOD_LABELS.get(method, method), fontsize=11)
            continue

        unique_d = np.sort(np.unique(dils))
        means = [float(imp[:, dils == d].mean()) for d in unique_d]
        stds = [float(imp[:, dils == d].std()) for d in unique_d]

        color = config.METHOD_COLORS.get(method, "#555555")
        ax.bar(range(len(unique_d)), means, yerr=stds,
               color=color, alpha=0.85, capsize=3,
               error_kw={'elinewidth': 1, 'capthick': 1})
        ax.set_xticks(range(len(unique_d)))
        ax.set_xticklabels([str(int(d)) for d in unique_d], fontsize=8)
        ax.set_xlabel("Dilation", fontsize=10)
        ax.set_ylabel("Mean Kernel Importance", fontsize=10)
        ax.set_title(f"{config.METHOD_LABELS.get(method, method)}", fontsize=11)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle(f"Kernel Importance by Dilation — "
                 f"{config.DB_META.get(db_key, {}).get('display_name', db_key)}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"figure_kernel_importance_combined_{db_key}.png")
    fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight", facecolor='white')
    plt.close(fig)
    print(f"  [SAVED] {save_path}", flush=True)

    return save_path

# ====================================================================
# 5. واجهة سطر الأوامر (CLI)
# ====================================================================
def parse_args():
    """
    تحليل معاملات سطر الأوامر.
    """
    parser = argparse.ArgumentParser(
        description="Paper 2 V2: Kernel Importance Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
أمثلة:
  # تشغيل التحليل الكامل
  python p05_analyze_kernel_importance.py

  # تخطي رسم الأشكال البيانية
  python p05_analyze_kernel_importance.py --skip_plot

  # تحديد مجلد الخصائص
  python p05_analyze_kernel_importance.py --features_dir ./path/to/features
        """
    )

    parser.add_argument("--features_dir", type=str, default=config.FEATURES_DIR,
                        help=f"مجلد الخصائص (الافتراضي {config.FEATURES_DIR})")

    parser.add_argument("--output_dir", type=str, default=config.TABLES_DIR,
                        help=f"مجلد جداول الإخراج (الافتراضي {config.TABLES_DIR})")

    parser.add_argument("--figures_dir", type=str, default=config.FIGURES_DIR,
                        help=f"مجلد الأشكال (الافتراضي {config.FIGURES_DIR})")

    parser.add_argument("--skip_plot", action="store_true",
                        help="تخطي رسم الأشكال البيانية")

    return parser.parse_args()

# ====================================================================
# 6. الدالة الرئيسية (Main)
# ====================================================================
def main():
    """
    النقطة الرئيسية لتشغيل السكربت.
    """
    args = parse_args()

    # ── طباعة معلومات البداية ──
    print("=" * 70, flush=True)
    print("  PAPER 2 V2: KERNEL IMPORTANCE ANALYSIS", flush=True)
    print("  Detach-ROCKET: Kernel Importance from Ridge Coefficients", flush=True)
    print("=" * 70, flush=True)
    print(f"  Features dir: {args.features_dir}", flush=True)
    print(f"  Output dir:   {args.output_dir}", flush=True)
    print(f"  Figures dir:  {args.figures_dir}", flush=True)
    print(f"  Skip plot:    {args.skip_plot}", flush=True)
    print("=" * 70 + "\n", flush=True)

    # ── 1. التحقق من وجود الملفات ──
    if not os.path.exists(args.features_dir):
        print(f"  [ERROR] Features directory not found: {args.features_dir}", flush=True)
        print("  Please run p01a_run_main_loso_benchmark.py first to generate kernel importance files.", flush=True)
        return

    # ── 2. معالجة كل قاعدة وطريقة ──
    for db_key in ["db7", "db3", "db2"]:
        print(f"\n{'='*70}", flush=True)
        print(f"  PROCESSING: {db_key}", flush=True)
        print(f"{'='*70}\n", flush=True)

        for method in config.METHODS:
            print(f"  Method: {method}", flush=True)

            imp, dils = load_kernel_importance(args.features_dir, f"{db_key}_{method}")
            if imp is None:
                print(f"    [SKIP] No data for {db_key}_{method}", flush=True)
                continue

            print(f"    {imp.shape[0]} folds, {imp.shape[1]} kernels", flush=True)

            # ── 2a. جدول أهمية كل نواة ──
            build_kernel_csv(db_key, method, imp, dils, args.output_dir)

            # ── 2b. جدول ملخص حسب التمدد ──
            build_dilation_summary_csv(db_key, method, imp, dils, args.output_dir)

            # ── 2c. شكل بياني (اختياري) ──
            if not args.skip_plot:
                plot_kernel_importance_by_dilation(args.features_dir, db_key, method, args.figures_dir)

        # ── 3. مقارنة قبل/بعد CORAL ──
        print(f"\n  Comparing raw vs coral for {db_key}...", flush=True)
        build_before_after_coral_csv(args.features_dir, db_key, args.output_dir)

        # ── 4. شكل بياني مدمج (اختياري) ──
        if not args.skip_plot:
            plot_combined_kernel_importance(args.features_dir, db_key, args.figures_dir)

    # ── 5. الملخص النهائي ──
    print("\n" + "=" * 70, flush=True)
    print("  KERNEL IMPORTANCE ANALYSIS COMPLETED!", flush=True)
    print(f"  Tables saved to: {args.output_dir}", flush=True)
    if not args.skip_plot:
        print(f"  Figures saved to: {args.figures_dir}", flush=True)
    print("=" * 70 + "\n", flush=True)

# ====================================================================
# 7. نقطة الدخول
# ====================================================================
if __name__ == "__main__":
    main()