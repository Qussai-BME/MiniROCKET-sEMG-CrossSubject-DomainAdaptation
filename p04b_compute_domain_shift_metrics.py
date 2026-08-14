"""
p04b_compute_domain_shift_metrics.py — Paper 2 V2: DOMAIN SHIFT METRICS
================================================================
هذا السكربت يقرأ ملفات JSON الناتجة عن p01a_run_main_loso_benchmark.py
ويستخرج مقاييس انحراف المجال (Covariance Frobenius و MMD) قبل وبعد التكيف.

المخرجات:
  1. domain_shift_metrics_{db}.csv — لكل قاعدة بيانات، يحتوي على المقاييس لكل شخص وطريقة.
  2. domain_shift_summary_{db}.csv — ملخص (المتوسط والانحراف المعياري) لكل طريقة ومقياس.
  3. figure_domain_shift_reduction.png — شكل بياني يوضح التخفيض في مسافة فروبينيوس.

الاستخدام:
  python p04b_compute_domain_shift_metrics.py
  python p04b_compute_domain_shift_metrics.py --skip_plot
  python p04b_compute_domain_shift_metrics.py --results_dir ./path/to/results
"""
import sys
import os
import json
import csv as _csv
import argparse
import numpy as np
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
# 2. دوال تحميل النتائج
# ====================================================================
def load_all_results(results_dir):
    """
    تحميل جميع ملفات JSON الناتجة عن p01a_run_main_loso_benchmark.py.

    المعاملات:
        results_dir (str): مجلد النتائج.

    المخرجات:
        dict: {db_key: {method: result_dict}}
    """
    all_data = {}

    if not os.path.exists(results_dir):
        print(f"  [WARN] Results directory not found: {results_dir}", flush=True)
        return all_data

    for fname in os.listdir(results_dir):
        if not fname.startswith("MiniROCKET_") or not fname.endswith("_results.json"):
            continue

        fpath = os.path.join(results_dir, fname)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                result = json.load(f)
        except Exception as e:
            print(f"  [WARN] Could not load {fname}: {e}", flush=True)
            continue

        db_key = result.get("database")
        method = result.get("method")

        if db_key is None or method is None:
            continue

        if db_key not in all_data:
            all_data[db_key] = {}
        all_data[db_key][method] = result

    return all_data

# ====================================================================
# 3. بناء جداول CSV
# ====================================================================
def build_domain_shift_csv(all_data, db_key, output_dir):
    """
    بناء جدول CSV لمقاييس انحراف المجال لقاعدة بيانات معينة.

    المعاملات:
        all_data (dict): جميع النتائج.
        db_key (str): مفتاح قاعدة البيانات.
        output_dir (str): مجلد الإخراج.

    المخرجات:
        str: مسار الملف المُنشأ.
    """
    methods = config.METHODS
    db_data = all_data.get(db_key, {})

    if not db_data:
        print(f"  [WARN] No data for {db_key}", flush=True)
        return None

    # ── 1. بناء الصفوف لكل شخص وطريقة ──
    rows = []
    for method in methods:
        res = db_data.get(method)
        if res is None:
            continue

        for subj in res.get("per_subject", []):
            ds = subj.get("domain_shift", {})
            rows.append({
                "database": db_key,
                "method": method,
                "subject_id": subj.get("subject_id"),
                "accuracy": subj.get("accuracy"),
                "cov_fro_before": ds.get("cov_frobenius_before"),
                "cov_fro_after": ds.get("cov_frobenius_after"),
                "mmd_before": ds.get("mmd_before"),
                "mmd_after": ds.get("mmd_after"),
            })

    if not rows:
        print(f"  [WARN] No domain shift data for {db_key}", flush=True)
        return None

    # ── 2. حفظ CSV التفصيلي ──
    os.makedirs(output_dir, exist_ok=True)
    detail_path = os.path.join(output_dir, f"domain_shift_metrics_{db_key}.csv")
    fieldnames = ["database", "method", "subject_id", "accuracy",
                  "cov_fro_before", "cov_fro_after",
                  "mmd_before", "mmd_after"]

    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [SAVED] {detail_path}", flush=True)

    # ── 3. حفظ CSV الملخص (متوسط لكل طريقة) ──
    summary_rows = []
    for method in methods:
        method_rows = [r for r in rows if r["method"] == method]
        if not method_rows:
            continue

        for metric in ["cov_fro_before", "cov_fro_after", "mmd_before", "mmd_after", "accuracy"]:
            vals = [r[metric] for r in method_rows if r[metric] is not None and not np.isnan(r[metric])]
            if not vals:
                continue

            # حساب نسبة التخفيض لـ covariance و mmd
            if metric == "cov_fro_before":
                after_vals = [r["cov_fro_after"] for r in method_rows if r["cov_fro_after"] is not None]
                if after_vals and len(vals) == len(after_vals):
                    reduction_vals = [(b - a) / b if b > 0 else 0 for b, a in zip(vals, after_vals)]
                    reduction_mean = np.mean(reduction_vals)
                    reduction_std = np.std(reduction_vals)
                else:
                    reduction_mean, reduction_std = np.nan, np.nan
            elif metric == "mmd_before":
                after_vals = [r["mmd_after"] for r in method_rows if r["mmd_after"] is not None]
                if after_vals and len(vals) == len(after_vals):
                    reduction_vals = [(b - a) / b if b > 0 else 0 for b, a in zip(vals, after_vals)]
                    reduction_mean = np.mean(reduction_vals)
                    reduction_std = np.std(reduction_vals)
                else:
                    reduction_mean, reduction_std = np.nan, np.nan
            else:
                reduction_mean, reduction_std = np.nan, np.nan

            summary_rows.append({
                "method": method,
                "metric": metric,
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0), 4),
                "median": round(float(np.median(vals)), 4),
                "n_subjects": len(vals),
                "reduction_mean": round(float(reduction_mean), 4) if not np.isnan(reduction_mean) else None,
                "reduction_std": round(float(reduction_std), 4) if not np.isnan(reduction_std) else None,
            })

    if summary_rows:
        summary_path = os.path.join(output_dir, f"domain_shift_summary_{db_key}.csv")
        fieldnames_summary = ["method", "metric", "mean", "std", "median", "n_subjects",
                              "reduction_mean", "reduction_std"]
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = _csv.DictWriter(f, fieldnames=fieldnames_summary, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"  [SAVED] {summary_path}", flush=True)

    return detail_path

# ====================================================================
# 4. رسم شكل بياني لانحراف المجال
# ====================================================================
def plot_domain_shift_reduction(all_data, output_dir):
    """
    رسم شكل بياني (بار شارت) يوضح مسافة فروبينيوس قبل وبعد التكيف.

    المعاملات:
        all_data (dict): جميع النتائج.
        output_dir (str): مجلد الإخراج.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not available. Skipping figure generation.", flush=True)
        return

    os.makedirs(output_dir, exist_ok=True)

    # إعداد الشكل: 3 رسوم بيانية (واحد لكل قاعدة)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

    for ax_idx, db_key in enumerate(["db7", "db3", "db2"]):
        ax = axes[ax_idx]
        db_data = all_data.get(db_key, {})

        if not db_data:
            ax.text(0.5, 0.5, f"No data for {db_key}", ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(config.DB_META.get(db_key, {}).get("display_name", db_key), fontsize=11)
            continue

        # جمع البيانات لكل طريقة
        methods_present = [m for m in config.METHODS if m in db_data]
        if not methods_present:
            ax.text(0.5, 0.5, f"No domain shift data for {db_key}", ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(config.DB_META.get(db_key, {}).get("display_name", db_key), fontsize=11)
            continue

        x = np.arange(len(methods_present))
        width = 0.35

        means_before = []
        means_after = []
        stds_before = []
        stds_after = []

        for method in methods_present:
            res = db_data[method]
            cov_before = []
            cov_after = []

            for subj in res.get("per_subject", []):
                ds = subj.get("domain_shift", {})
                cb = ds.get("cov_frobenius_before")
                ca = ds.get("cov_frobenius_after")
                if cb is not None and not np.isnan(cb):
                    cov_before.append(cb)
                if ca is not None and not np.isnan(ca):
                    cov_after.append(ca)

            means_before.append(np.mean(cov_before) if cov_before else 0)
            means_after.append(np.mean(cov_after) if cov_after else 0)
            stds_before.append(np.std(cov_before, ddof=1) if len(cov_before) > 1 else 0)
            stds_after.append(np.std(cov_after, ddof=1) if len(cov_after) > 1 else 0)

        # رسم الأشرطة
        # ملاحظة مهمة: TCA يعمل في فضاء ميزات مختلف كلياً (بعد إسقاط
        # eigenvalue) بقيم أكبر بمراتب عن raw/centering/coral/SA التي
        # تبقى قريبة من فضاء PPV الأصلي [0,1]. على مقياس خطي مشترك،
        # هذا يجعل باقي الطرق تبدو صفراً بصرياً. نستخدم مقياس لوغاريتمي
        # ليظهر الجميع بوضوح، مع رفع القيم الصفرية لحد أدنى صغير جداً
        # (فقط لغرض الرسم؛ لا يؤثر على القيم المحفوظة في CSV).
        eps = 1e-3
        means_before_plot = [max(v, eps) for v in means_before]
        means_after_plot  = [max(v, eps) for v in means_after]

        bars1 = ax.bar(x - width/2, means_before_plot, width,
                       yerr=stds_before, label="Before",
                       color="#E53935", alpha=0.8, capsize=3,
                       error_kw={'elinewidth': 1, 'capthick': 1})
        bars2 = ax.bar(x + width/2, means_after_plot, width,
                       yerr=stds_after, label="After",
                       color="#43A047", alpha=0.8, capsize=3,
                       error_kw={'elinewidth': 1, 'capthick': 1})
        ax.set_yscale('log')

        ax.set_xticks(x)
        ax.set_xticklabels([config.METHOD_LABELS.get(m, m) for m in methods_present],
                           rotation=30, ha="right", fontsize=9)
        ax.set_title(config.DB_META.get(db_key, {}).get("display_name", db_key), fontsize=11)
        ax.set_ylabel("Covariance Frobenius Distance (log scale)", fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(axis='y', alpha=0.3, which='both')

    fig.suptitle("Domain Shift Reduction: Covariance Distance Before/After Adaptation",
                 fontsize=13, fontweight="bold", y=1.02)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "figure_domain_shift_reduction.png")
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
        description="Paper 2 V2: Domain Shift Metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
أمثلة:
  # تشغيل التحليل الكامل
  python p04b_compute_domain_shift_metrics.py

  # تخطي رسم الشكل البياني
  python p04b_compute_domain_shift_metrics.py --skip_plot

  # تحديد مجلد النتائج
  python p04b_compute_domain_shift_metrics.py --results_dir ./my_results
        """
    )

    parser.add_argument("--results_dir", type=str, default=config.RESULTS_DIR,
                        help=f"مجلد النتائج (الافتراضي {config.RESULTS_DIR})")

    parser.add_argument("--output_dir", type=str, default=config.TABLES_DIR,
                        help=f"مجلد الإخراج (الافتراضي {config.TABLES_DIR})")

    parser.add_argument("--figures_dir", type=str, default=config.FIGURES_DIR,
                        help=f"مجلد الأشكال (الافتراضي {config.FIGURES_DIR})")

    parser.add_argument("--skip_plot", action="store_true",
                        help="تخطي رسم الشكل البياني")

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
    print("  PAPER 2 V2: DOMAIN SHIFT METRICS", flush=True)
    print("  Extracting Covariance Frobenius and MMD metrics", flush=True)
    print("=" * 70, flush=True)
    print(f"  Results dir: {args.results_dir}", flush=True)
    print(f"  Output dir:  {args.output_dir}", flush=True)
    print(f"  Figures dir: {args.figures_dir}", flush=True)
    print(f"  Skip plot:   {args.skip_plot}", flush=True)
    print("=" * 70 + "\n", flush=True)

    # ── 1. تحميل النتائج ──
    print("  Loading results...", flush=True)
    all_data = load_all_results(args.results_dir)

    if not all_data:
        print("  [ERROR] No results found!", flush=True)
        return

    print(f"  Found data for DBs: {list(all_data.keys())}", flush=True)

    # ── 2. بناء جداول CSV لكل قاعدة ──
    print("\n" + "=" * 70, flush=True)
    print("  BUILDING DOMAIN SHIFT TABLES", flush=True)
    print("=" * 70 + "\n", flush=True)

    for db_key in ["db7", "db3", "db2"]:
        build_domain_shift_csv(all_data, db_key, args.output_dir)

    # ── 3. رسم الشكل البياني (اختياري) ──
    if not args.skip_plot:
        print("\n" + "=" * 70, flush=True)
        print("  GENERATING FIGURE", flush=True)
        print("=" * 70 + "\n", flush=True)

        plot_domain_shift_reduction(all_data, args.figures_dir)

    # ── 4. الملخص النهائي ──
    print("\n" + "=" * 70, flush=True)
    print("  DOMAIN SHIFT METRICS COMPLETED!", flush=True)
    print(f"  Tables saved to: {args.output_dir}", flush=True)
    if not args.skip_plot:
        print(f"  Figure saved to: {args.figures_dir}", flush=True)
    print("=" * 70 + "\n", flush=True)

# ====================================================================
# 7. نقطة الدخول
# ====================================================================
if __name__ == "__main__":
    main()