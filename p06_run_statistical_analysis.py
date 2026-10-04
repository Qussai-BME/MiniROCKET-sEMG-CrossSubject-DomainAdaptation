"""
p06_run_statistical_analysis.py — Paper 2 V2: COMPREHENSIVE STATISTICAL ANALYSIS
====================================================================
هذا السكربت هو المسؤول عن جميع التحليلات الإحصائية النهائية للورقة.

يقرأ هذا السكربت ملفات JSON الناتجة عن:
  - p01a_run_main_loso_benchmark.py (النتائج الرئيسية: 5 طرق × 3 قواعد)
  - p02_run_window_and_sample_ablation.py (نتائج الإزالة: حجم النافذة، عدد النوى، إلخ)

ثم يقوم بتطبيق الاختبارات الإحصائية التالية:
  1. اختبار فريدمان (مع تصحيح إيمان-دافنبورت)
  2. اختبار ويلكوكسون للمقارنات الزوجية
  3. تصحيح هولم-سيداك للمقارنات المتعددة
  4. اختبار نيميني (المسافة الحرجة)
  5. حجم التأثير كوهين د
  6. تحليل F1 لكل صنف
  7. تحليل أوقات التشغيل

المخرجات (لكل قاعدة من القواعد الثلاث):
  - Table2_main_results.csv + .tex
  - TableS1_per_subject.csv + .tex
  - TableS2_wilcoxon.csv + .tex
  - TableS3_friedman.csv + .tex
  - TableS4_holmsidak.csv + .tex
  - TableS5_nemenyi.csv + .tex
  - TableS6_perclass_f1.csv + .tex
  - TableS7_timing.csv + .tex
  - TableS8_cohend.csv + .tex
  - ALL_SUMMARY.txt

الاستخدام:
  python p06_run_statistical_analysis.py
  python p06_run_statistical_analysis.py --results_dir ./path/to/results
  python p06_run_statistical_analysis.py --skip_ablation
"""
import sys
import os
import json
import csv as _csv
import argparse
import numpy as np
import warnings
from datetime import datetime
from itertools import combinations

# ====================================================================
# 1. إعدادات المسار والاستيرادات
# ====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import config
from statistical_utils import (
    per_class_f1,
    confusion_matrix,
    descriptive_stats,
    friedman_test,
    nemenyi_posthoc,
    wilcoxon_pairwise,
    holm_sidak_correction,
    cohens_d,
    interpret_cohens_d,
    random_baseline_from_labels,
    ratio_vs_random,
    comprehensive_statistical_analysis,
)

# تجاهل التحذيرات
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ====================================================================
# 2. دوال تحميل النتائج
# ====================================================================
def load_results_from_dir(results_dir):
    """
    تحميل جميع ملفات JSON من مجلد النتائج.

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

def _derive_missing_summary_fields(res):
    """
    FIX-P06-COMPAT (اكتُشف أثناء إعادة تشغيل p06 على نتائج DB3/DB7
    الحقيقية بعد التدقيق): هذا الملف كُتب أصلاً لمخطط نتائج V2 القديم،
    الذي كان يضع n_classes على المستوى الأعلى للنتيجة، وavg_random_
    baseline_pct/avg_ratio_vs_random داخل summary مباشرة. مخطط V3
    الجديد (بعد بنود التدقيق) لا يضع أياً من هذه الثلاثة في هذين
    الموقعين — فكان `res.get('n_classes', 0)` يعيد 0 دائماً،
    و`summary.get('avg_random_baseline_pct', np.nan)` يعيد NaN دائماً،
    مما يُسبب إما قيماً صامتة خاطئة (0.0% في ALL_SUMMARY.txt) أو
    ValueError عند تنسيق '—' كـ float في LaTeX (Table 2).

    الإصلاح: اشتقاق الثلاثة من per_subject مباشرة (موجودة هناك في كلا
    المخططين) إن غابت عن مواقعها القديمة، بدل الصمت على غيابها.
    يُعدَّل res['summary'] في مكانه، ويُعاد n_classes المُشتقّ.
    """
    per_subject = res.get('per_subject', [])
    summary = res.setdefault('summary', {})

    if summary.get('avg_random_baseline_pct') in (None,) or \
            'avg_random_baseline_pct' not in summary:
        vals = [s['random_baseline_pct'] for s in per_subject
                if s.get('random_baseline_pct') is not None]
        summary['avg_random_baseline_pct'] = float(np.mean(vals)) if vals else np.nan

    if summary.get('avg_ratio_vs_random') in (None,) or \
            'avg_ratio_vs_random' not in summary:
        vals = [s['ratio_vs_random'] for s in per_subject
                if s.get('ratio_vs_random') is not None]
        summary['avg_ratio_vs_random'] = float(np.mean(vals)) if vals else np.nan

    n_classes = res.get('n_classes', 0)
    if not n_classes:
        for s in per_subject:
            fpc = s.get('f1_per_class')
            if fpc:
                n_classes = len(fpc)
                break
            cm = s.get('confusion_matrix')
            if cm:
                n_classes = len(cm)
                break
    return n_classes


def load_ablation_results(ablation_dir):
    """
    تحميل نتائج الإزالة من مجلد الإزالة.

    المعاملات:
        ablation_dir (str): مجلد الإزالة.

    المخرجات:
        dict: {ablation_type: {label: result_dict}}
    """
    all_data = {
        'window': {},
        'kernel': {},
        'coral_reg': {},
        'grouping': {},
    }

    if not os.path.exists(ablation_dir):
        print(f"  [WARN] Ablation directory not found: {ablation_dir}", flush=True)
        return all_data

    for fname in os.listdir(ablation_dir):
        if not fname.startswith("MiniROCKET_") or not fname.endswith("_results.json"):
            continue

        fpath = os.path.join(ablation_dir, fname)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                result = json.load(f)
        except Exception:
            continue

        # تحديد نوع الإزالة من اسم الملف
        if "window" in fname:
            all_data['window'][fname] = result
        elif "kernels" in fname:
            all_data['kernel'][fname] = result
        elif "reg" in fname:
            all_data['coral_reg'][fname] = result
        elif "grouped" in fname:
            all_data['grouping'][fname] = result

    return all_data

# ====================================================================
# 3. دوال حفظ الجداول (CSV و LaTeX)
# ====================================================================
def save_csv(filepath, rows, fieldnames):
    """
    حفظ قائمة من القواميس في ملف CSV.

    المعاملات:
        filepath (str): مسار الملف.
        rows (list): قائمة من القواميس.
        fieldnames (list): أسماء الأعمدة.
    """
    if not rows:
        return

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = _csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [SAVED] {filepath}", flush=True)

def save_latex(filepath, content):
    """
    حفظ محتوى LaTeX في ملف.

    المعاملات:
        filepath (str): مسار الملف.
        content (str): محتوى LaTeX.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  [SAVED] {filepath}", flush=True)

# ====================================================================
# 4. توليد الجداول الإحصائية
# ====================================================================
def generate_main_results_table(db_data, db_key, output_dir):
    """
    توليد جدول النتائج الرئيسية (Table 2).

    المعاملات:
        db_data (dict): نتائج قاعدة البيانات.
        db_key (str): مفتاح قاعدة البيانات.
        output_dir (str): مجلد الإخراج.

    المخرجات:
        str: مسار الملف المُنشأ.
    """
    methods = config.METHODS
    rows = []

    for method in methods:
        res = db_data.get(method)
        if res is None:
            continue

        summary = res.get('summary', {})
        n_subjects = res.get('n_subjects', 0)
        n_classes = _derive_missing_summary_fields(res)

        rows.append({
            'database': db_key,
            'method': method,
            'method_label': config.METHOD_LABELS.get(method, method),
            'n_subjects': n_subjects,
            'n_classes': n_classes,
            'accuracy_mean': summary.get('mean', np.nan),
            'accuracy_std': summary.get('std', np.nan),
            'accuracy_median': summary.get('median', np.nan),
            'accuracy_min': summary.get('min', np.nan),
            'accuracy_max': summary.get('max', np.nan),
            'ci95_lower': summary.get('ci95_lower', np.nan),
            'ci95_upper': summary.get('ci95_upper', np.nan),
            'avg_random_baseline': summary.get('avg_random_baseline_pct', np.nan),
            'avg_ratio_vs_random': summary.get('avg_ratio_vs_random', np.nan),
        })

    if not rows:
        return None

    # CSV
    csv_path = os.path.join(output_dir, f'Table2_main_results_{db_key}.csv')
    fieldnames = ['database', 'method', 'method_label', 'n_subjects', 'n_classes',
                  'accuracy_mean', 'accuracy_std', 'accuracy_median', 'accuracy_min', 'accuracy_max',
                  'ci95_lower', 'ci95_upper', 'avg_random_baseline', 'avg_ratio_vs_random']
    save_csv(csv_path, rows, fieldnames)

    # LaTeX
    tex_content = generate_main_results_latex(rows, db_key)
    tex_path = os.path.join(output_dir, f'Table2_main_results_{db_key}.tex')
    save_latex(tex_path, tex_content)

    return csv_path

def generate_main_results_latex(rows, db_key):
    """
    توليد محتوى LaTeX لجدول النتائج الرئيسية.
    """
    db_display = config.DB_META.get(db_key, {}).get('display_name', db_key)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{LOSO Cross-Validation Results — " + db_display + r"}",
        r"\label{tab:main_results_" + db_key + r"}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Method & Accuracy (\%) & Std (\%) & Median (\%) & Min (\%) & Max (\%) & Random Baseline \\",
        r"\midrule",
    ]

    for row in rows:
        method = row['method_label']
        acc = row['accuracy_mean'] * 100 if not np.isnan(row['accuracy_mean']) else '—'
        std = row['accuracy_std'] * 100 if not np.isnan(row['accuracy_std']) else '—'
        med = row['accuracy_median'] * 100 if not np.isnan(row['accuracy_median']) else '—'
        mn = row['accuracy_min'] * 100 if not np.isnan(row['accuracy_min']) else '—'
        mx = row['accuracy_max'] * 100 if not np.isnan(row['accuracy_max']) else '—'
        rand = row['avg_random_baseline'] if not np.isnan(row['avg_random_baseline']) else '—'

        # تحديد أفضل طريقة (أعلى متوسط دقة)
        acc_vals = [r['accuracy_mean'] for r in rows if not np.isnan(r['accuracy_mean'])]
        if acc_vals and row['accuracy_mean'] == max(acc_vals):
            method = r"\textbf{" + method + r"}"

        lines.append(f"{method} & {acc if isinstance(acc, str) else f'{acc:.2f}'} & "
                     f"{std if isinstance(std, str) else f'{std:.2f}'} & "
                     f"{med if isinstance(med, str) else f'{med:.2f}'} & "
                     f"{mn if isinstance(mn, str) else f'{mn:.2f}'} & "
                     f"{mx if isinstance(mx, str) else f'{mx:.2f}'} & "
                     f"{rand if isinstance(rand, str) else f'{rand:.1f}'}\\% \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
    ])

    return "\n".join(lines)

def generate_per_subject_table(db_data, db_key, output_dir):
    """
    توليد جدول النتائج لكل شخص (Table S1).

    المعاملات:
        db_data (dict): نتائج قاعدة البيانات.
        db_key (str): مفتاح قاعدة البيانات.
        output_dir (str): مجلد الإخراج.

    المخرجات:
        str: مسار الملف المُنشأ.
    """
    methods = config.METHODS

    # جمع البيانات لكل شخص
    subject_data = {}
    for method in methods:
        res = db_data.get(method)
        if res is None:
            continue

        for subj in res.get('per_subject', []):
            sid = subj.get('subject_id')
            if sid is None:
                continue
            if sid not in subject_data:
                subject_data[sid] = {}
            subject_data[sid][method] = {
                'accuracy': subj.get('accuracy', np.nan),
                'random_baseline': subj.get('random_baseline_pct', np.nan),
                'ratio': subj.get('ratio_vs_random', np.nan),
            }

    if not subject_data:
        return None

    # بناء الصفوف
    rows = []
    for sid in sorted(subject_data.keys()):
        row = {'subject_id': sid}
        for method in methods:
            if method in subject_data[sid]:
                row[f'{method}_accuracy'] = subject_data[sid][method].get('accuracy', np.nan)
                row[f'{method}_random'] = subject_data[sid][method].get('random_baseline', np.nan)
                row[f'{method}_ratio'] = subject_data[sid][method].get('ratio', np.nan)
            else:
                row[f'{method}_accuracy'] = np.nan
                row[f'{method}_random'] = np.nan
                row[f'{method}_ratio'] = np.nan
        rows.append(row)

    # CSV
    csv_path = os.path.join(output_dir, f'TableS1_per_subject_{db_key}.csv')
    fieldnames = ['subject_id'] + [f'{m}_accuracy' for m in methods] + \
                 [f'{m}_random' for m in methods] + [f'{m}_ratio' for m in methods]
    save_csv(csv_path, rows, fieldnames)

    return csv_path

def generate_wilcoxon_table(db_data, db_key, output_dir):
    """
    توليد جدول اختبار ويلكوكسون (Table S2).

    المعاملات:
        db_data (dict): نتائج قاعدة البيانات.
        db_key (str): مفتاح قاعدة البيانات.
        output_dir (str): مجلد الإخراج.

    المخرجات:
        str: مسار الملف المُنشأ.
    """
    # Centering is excluded from inferential statistics (identical to Raw).
    # من raw بعد StandardScaler) — يبقى في الجداول الوصفية (Table 2) فقط.
    methods = list(config.INFERENTIAL_METHODS)

    # FIX-P06-COMPAT (نفس الاكتشاف أثناء إعادة التشغيل الفعلي): هذه
    # الدالة كانت تشير إلى `wilcoxon_results` دون أي تعريف سابق له في
    # النطاق (NameError) — يبدو أنه سطر حساب مفقود من تعديل سابق. يُعاد
    # بناؤه هنا بنفس نمط مصفوفة الدقة المستخدم في generate_friedman_table
    # أدناه، ثم استدعاء wilcoxon_pairwise() الموجودة أصلاً في
    # statistical_utils.py (لم تُعدَّل، وتدعم أي عدد طرق أصلاً).
    subject_ids = []
    for method in methods:
        res = db_data.get(method)
        if res is not None:
            subject_ids = [s['subject_id'] for s in res.get('per_subject', [])]
            break

    if not subject_ids:
        return None

    acc_matrix = []
    for method in methods:
        res = db_data.get(method)
        if res is None:
            acc_col = [np.nan] * len(subject_ids)
        else:
            acc_map = {s['subject_id']: s.get('accuracy', np.nan) for s in res.get('per_subject', [])}
            acc_col = [acc_map.get(sid, np.nan) for sid in subject_ids]
        acc_matrix.append(acc_col)
    acc_matrix = np.array(acc_matrix).T  # (n_subjects, n_methods)

    wilcoxon_results = wilcoxon_pairwise(acc_matrix, methods)

    rows = []
    for w in wilcoxon_results:
        rows.append({
            'method_1': w['method_1'],
            'method_2': w['method_2'],
            'statistic': w['statistic'],
            'p_value': w['p_value'],
            'significant': w['significant'],
            'n': w['n'],
            'effect_size_r': w.get('effect_size_r', np.nan),
            'note': w.get('note', ''),
        })

    # CSV
    csv_path = os.path.join(output_dir, f'TableS2_wilcoxon_{db_key}.csv')
    fieldnames = ['method_1', 'method_2', 'statistic', 'p_value', 'significant', 'n', 'effect_size_r', 'note']
    save_csv(csv_path, rows, fieldnames)

    # LaTeX
    tex_content = generate_wilcoxon_latex(rows)
    tex_path = os.path.join(output_dir, f'TableS2_wilcoxon_{db_key}.tex')
    save_latex(tex_path, tex_content)

    return csv_path

def generate_wilcoxon_latex(rows):
    """
    توليد محتوى LaTeX لجدول ويلكوكسون.
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Wilcoxon Signed-Rank Test Pairwise Comparisons}",
        r"\label{tab:wilcoxon}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Comparison & W & p-value & Effect Size (r) & Sig. \\",
        r"\midrule",
    ]

    for row in rows:
        comp = f"{row['method_1']} vs {row['method_2']}"
        W = f"{row['statistic']:.0f}" if not np.isnan(row['statistic']) else '—'
        p = f"{row['p_value']:.4f}" if not np.isnan(row['p_value']) else '—'
        r = f"{row['effect_size_r']:.3f}" if not np.isnan(row['effect_size_r']) else '—'
        sig = r"$^{***}$" if row.get('significant') and row.get('p_value', 1) < 0.001 else \
              r"$^{**}$" if row.get('significant') and row.get('p_value', 1) < 0.01 else \
              r"$^{*}$" if row.get('significant') else "n.s."

        lines.append(f"{comp} & {W} & {p} & {r} & {sig} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)

def generate_friedman_table(db_data, db_key, output_dir):
    """
    توليد جدول اختبار فريدمان (Table S3).

    المعاملات:
        db_data (dict): نتائج قاعدة البيانات.
        db_key (str): مفتاح قاعدة البيانات.
        output_dir (str): مجلد الإخراج.

    المخرجات:
        str: مسار الملف المُنشأ.
    """
    # Centering is excluded from inferential statistics (identical to Raw).
    methods = list(config.INFERENTIAL_METHODS)

    # بناء مصفوفة الدقة
    acc_matrix = []
    subject_ids = []

    for method in methods:
        res = db_data.get(method)
        if res is not None:
            subject_ids = [s['subject_id'] for s in res.get('per_subject', [])]
            break

    if not subject_ids:
        return None

    for method in methods:
        res = db_data.get(method)
        if res is None:
            acc_col = [np.nan] * len(subject_ids)
        else:
            acc_map = {s['subject_id']: s.get('accuracy', np.nan) for s in res.get('per_subject', [])}
            acc_col = [acc_map.get(sid, np.nan) for sid in subject_ids]
        acc_matrix.append(acc_col)

    acc_matrix = np.array(acc_matrix).T  # (n_subjects, n_methods)

    # إزالة الأشخاص الذين لديهم قيم مفقودة
    valid_mask = ~np.isnan(acc_matrix).any(axis=1)
    acc_matrix_clean = acc_matrix[valid_mask]

    if acc_matrix_clean.shape[0] < 3:
        return None

    # اختبار فريدمان
    friedman = friedman_test(acc_matrix_clean, method_names=methods)

    rows = [{
        'database': db_key,
        'n_subjects': friedman['n_subjects'],
        'n_methods': friedman['n_methods'],
        'chi2_statistic': friedman['statistic'],
        'p_value': friedman['p_value'],
        'p_chi2': friedman.get('p_chi2', np.nan),
        'f_iman': friedman.get('f_iman', np.nan),
        'p_iman': friedman.get('p_iman', np.nan),
        'significant': friedman['p_value'] < 0.05 if not np.isnan(friedman['p_value']) else False,
    }]

    # إضافة متوسط الرتب لكل طريقة
    for i, method in enumerate(methods):
        rows[0][f'rank_{method}'] = friedman['mean_ranks'][i] if i < len(friedman['mean_ranks']) else np.nan

    # CSV
    csv_path = os.path.join(output_dir, f'TableS3_friedman_{db_key}.csv')
    fieldnames = ['database', 'n_subjects', 'n_methods', 'chi2_statistic', 'p_value',
                  'p_chi2', 'f_iman', 'p_iman', 'significant'] + [f'rank_{m}' for m in methods]
    save_csv(csv_path, rows, fieldnames)

    # LaTeX
    tex_content = generate_friedman_latex(rows[0], methods, db_key)
    tex_path = os.path.join(output_dir, f'TableS3_friedman_{db_key}.tex')
    save_latex(tex_path, tex_content)

    return csv_path

def generate_friedman_latex(row, methods, db_key):
    """
    توليد محتوى LaTeX لجدول فريدمان.
    """
    db_display = config.DB_META.get(db_key, {}).get('display_name', db_key)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Friedman Test — " + db_display + r"}",
        r"\label{tab:friedman_" + db_key + r"}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Statistic & Value & df & p-value & Significance \\",
        r"\midrule",
    ]

    chi2 = f"{row['chi2_statistic']:.3f}" if not np.isnan(row['chi2_statistic']) else '—'
    p = f"{row['p_value']:.4f}" if not np.isnan(row['p_value']) else '—'
    df = row['n_methods'] - 1
    sig = r"$^{***}$" if row['p_value'] < 0.001 else \
          r"$^{**}$" if row['p_value'] < 0.01 else \
          r"$^{*}$" if row['p_value'] < 0.05 else "n.s."

    lines.append(rf"Friedman $\chi^2$ & {chi2} & {df} & {p} & {sig} \\")
    lines.append(r"\midrule")

    # متوسط الرتب
    lines.append(r"\multicolumn{5}{c}{Mean Ranks} \\")
    lines.append(r"\midrule")
    for method in methods:
        rank = row.get(f'rank_{method}', np.nan)
        rank_str = f"{rank:.2f}" if not np.isnan(rank) else '—'
        lines.append(f"{method} & & & & {rank_str} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)

def generate_holmsidak_table(db_data, db_key, output_dir):
    """
    توليد جدول تصحيح هولم-سيداك (Table S4).

    المعاملات:
        db_data (dict): نتائج قاعدة البيانات.
        db_key (str): مفتاح قاعدة البيانات.
        output_dir (str): مجلد الإخراج.

    المخرجات:
        str: مسار الملف المُنشأ.
    """
    # Centering is excluded from inferential statistics (identical to Raw).
    methods = list(config.INFERENTIAL_METHODS)

    # بناء مصفوفة الدقة
    acc_matrix = []
    subject_ids = []

    for method in methods:
        res = db_data.get(method)
        if res is not None:
            subject_ids = [s['subject_id'] for s in res.get('per_subject', [])]
            break

    if not subject_ids:
        return None

    for method in methods:
        res = db_data.get(method)
        if res is None:
            acc_col = [np.nan] * len(subject_ids)
        else:
            acc_map = {s['subject_id']: s.get('accuracy', np.nan) for s in res.get('per_subject', [])}
            acc_col = [acc_map.get(sid, np.nan) for sid in subject_ids]
        acc_matrix.append(acc_col)

    acc_matrix = np.array(acc_matrix).T

    # إزالة الأشخاص الذين لديهم قيم مفقودة
    valid_mask = ~np.isnan(acc_matrix).any(axis=1)
    acc_matrix_clean = acc_matrix[valid_mask]

    if acc_matrix_clean.shape[0] < 3:
        return None

    # اختبار ويلكوكسون
    wilcoxon_results = wilcoxon_pairwise(acc_matrix_clean, method_names=methods)

    # استخراج قيم p
    p_values = [w['p_value'] for w in wilcoxon_results if not np.isnan(w['p_value'])]

    if not p_values:
        return None

    # تصحيح هولم-سيداك
    corrected = holm_sidak_correction(p_values)

    rows = []
    for w, c in zip(wilcoxon_results, corrected):
        rows.append({
            'method_1': w['method_1'],
            'method_2': w['method_2'],
            'raw_p': c['raw_p'],
            'adjusted_p': c['adjusted_p'],
            'significant': c['significant'],
            'rejected': c['rejected'],
        })

    # CSV
    csv_path = os.path.join(output_dir, f'TableS4_holmsidak_{db_key}.csv')
    fieldnames = ['method_1', 'method_2', 'raw_p', 'adjusted_p', 'significant', 'rejected']
    save_csv(csv_path, rows, fieldnames)

    # LaTeX
    tex_content = generate_holmsidak_latex(rows)
    tex_path = os.path.join(output_dir, f'TableS4_holmsidak_{db_key}.tex')
    save_latex(tex_path, tex_content)

    return csv_path

def generate_holmsidak_latex(rows):
    """
    توليد محتوى LaTeX لجدول هولم-سيداك.
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Holm-Sidak Correction for Multiple Comparisons}",
        r"\label{tab:holmsidak}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Comparison & Raw p & Adjusted p & Rejected \\",
        r"\midrule",
    ]

    for row in rows:
        comp = f"{row['method_1']} vs {row['method_2']}"
        raw = f"{row['raw_p']:.4f}" if not np.isnan(row['raw_p']) else '—'
        adj = f"{row['adjusted_p']:.4f}" if not np.isnan(row['adjusted_p']) else '—'
        rej = "Yes" if row['rejected'] else "No"

        lines.append(f"{comp} & {raw} & {adj} & {rej} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)

def generate_nemenyi_table(db_data, db_key, output_dir):
    """
    توليد جدول اختبار نيميني (Table S5).

    المعاملات:
        db_data (dict): نتائج قاعدة البيانات.
        db_key (str): مفتاح قاعدة البيانات.
        output_dir (str): مجلد الإخراج.

    المخرجات:
        str: مسار الملف المُنشأ.
    """
    # Centering is excluded from inferential statistics (identical to Raw).
    methods = list(config.INFERENTIAL_METHODS)

    # بناء مصفوفة الدقة
    acc_matrix = []
    subject_ids = []

    for method in methods:
        res = db_data.get(method)
        if res is not None:
            subject_ids = [s['subject_id'] for s in res.get('per_subject', [])]
            break

    if not subject_ids:
        return None

    for method in methods:
        res = db_data.get(method)
        if res is None:
            acc_col = [np.nan] * len(subject_ids)
        else:
            acc_map = {s['subject_id']: s.get('accuracy', np.nan) for s in res.get('per_subject', [])}
            acc_col = [acc_map.get(sid, np.nan) for sid in subject_ids]
        acc_matrix.append(acc_col)

    acc_matrix = np.array(acc_matrix).T

    # إزالة الأشخاص الذين لديهم قيم مفقودة
    valid_mask = ~np.isnan(acc_matrix).any(axis=1)
    acc_matrix_clean = acc_matrix[valid_mask]

    if acc_matrix_clean.shape[0] < 3 or acc_matrix_clean.shape[1] < 3:
        return None

    # اختبار فريدمان أولاً
    friedman = friedman_test(acc_matrix_clean, method_names=methods)

    if np.isnan(friedman['p_value']) or friedman['p_value'] >= 0.05:
        return None

    # اختبار نيميني
    nemenyi = nemenyi_posthoc(
        friedman['mean_ranks'],
        friedman['n_subjects'],
        friedman['n_methods'],
        alpha=0.05
    )

    rows = []
    for comp in nemenyi['comparisons']:
        rows.append({
            'method_i': methods[comp['method_i']],
            'method_j': methods[comp['method_j']],
            'rank_diff': comp['rank_diff'],
            'critical_distance': comp['critical_difference'],
            'significant': comp['significant'],
            'p_approx': comp.get('p_approx', np.nan),
        })

    # إضافة المسافة الحرجة كصف منفصل
    rows.append({
        'method_i': 'CD',
        'method_j': str(nemenyi['critical_difference']),
        'rank_diff': np.nan,
        'critical_distance': nemenyi['critical_difference'],
        'significant': None,
        'p_approx': np.nan,
    })

    # CSV
    csv_path = os.path.join(output_dir, f'TableS5_nemenyi_{db_key}.csv')
    fieldnames = ['method_i', 'method_j', 'rank_diff', 'critical_distance', 'significant', 'p_approx']
    save_csv(csv_path, rows, fieldnames)

    # LaTeX
    tex_content = generate_nemenyi_latex(rows, methods)
    tex_path = os.path.join(output_dir, f'TableS5_nemenyi_{db_key}.tex')
    save_latex(tex_path, tex_content)

    return csv_path

def generate_nemenyi_latex(rows, methods):
    """
    توليد محتوى LaTeX لجدول نيميني.
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Nemenyi Post-hoc Test Pairwise Comparisons}",
        r"\label{tab:nemenyi}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Comparison & Rank Difference & Critical Distance & Significant \\",
        r"\midrule",
    ]

    for row in rows:
        if row['method_i'] == 'CD':
            continue
        comp = f"{row['method_i']} vs {row['method_j']}"
        diff = f"{row['rank_diff']:.3f}" if not np.isnan(row['rank_diff']) else '—'
        cd = f"{row['critical_distance']:.3f}" if not np.isnan(row['critical_distance']) else '—'
        sig = "Yes" if row['significant'] else "No"

        lines.append(f"{comp} & {diff} & {cd} & {sig} \\\\")

    # إضافة المسافة الحرجة
    cd_row = next((r for r in rows if r['method_i'] == 'CD'), None)
    if cd_row:
        lines.append(r"\midrule")
        lines.append(f"Critical Distance (CD) & & {cd_row['critical_distance']:.3f} & \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)

def generate_perclass_f1_table(db_data, db_key, output_dir):
    """
    توليد جدول F1 لكل صنف (Table S6).

    المعاملات:
        db_data (dict): نتائج قاعدة البيانات.
        db_key (str): مفتاح قاعدة البيانات.
        output_dir (str): مجلد الإخراج.

    المخرجات:
        str: مسار الملف المُنشأ.
    """
    methods = config.METHODS

    # تحديد أفضل طريقة (أعلى متوسط دقة)
    best_method = None
    best_acc = -1
    for method in methods:
        res = db_data.get(method)
        if res is not None:
            acc = res.get('summary', {}).get('mean', 0)
            if acc > best_acc:
                best_acc = acc
                best_method = method

    if best_method is None:
        return None

    # الحصول على F1 لكل صنف من أفضل طريقة ومن raw
    best_res = db_data.get(best_method)
    raw_res = db_data.get('raw')

    if best_res is None:
        return None

    # جمع F1 لكل شخص
    best_f1s = []
    raw_f1s = []

    for subj in best_res.get('per_subject', []):
        f1_list = subj.get('f1_per_class', [])
        if f1_list:
            best_f1s.append(f1_list)

    if raw_res is not None:
        for subj in raw_res.get('per_subject', []):
            f1_list = subj.get('f1_per_class', [])
            if f1_list:
                raw_f1s.append(f1_list)

    if not best_f1s:
        return None

    # حساب متوسط F1 لكل صنف
    n_classes = len(best_f1s[0])
    best_mean = np.mean(best_f1s, axis=0)
    best_std = np.std(best_f1s, axis=0, ddof=1) if len(best_f1s) > 1 else np.zeros(n_classes)

    if raw_f1s:
        raw_mean = np.mean(raw_f1s, axis=0)
        raw_std = np.std(raw_f1s, axis=0, ddof=1) if len(raw_f1s) > 1 else np.zeros(n_classes)
    else:
        raw_mean = np.zeros(n_classes)
        raw_std = np.zeros(n_classes)

    # بناء الصفوف
    rows = []
    for c in range(n_classes):
        rows.append({
            'class_id': c,
            'class_name': f'Class {c}',
            f'{best_method}_f1_mean': best_mean[c],
            f'{best_method}_f1_std': best_std[c],
            'raw_f1_mean': raw_mean[c] if raw_f1s else np.nan,
            'raw_f1_std': raw_std[c] if raw_f1s else np.nan,
            'improvement': (best_mean[c] - raw_mean[c]) if raw_f1s else np.nan,
        })

    # CSV
    csv_path = os.path.join(output_dir, f'TableS6_perclass_f1_{db_key}.csv')
    fieldnames = ['class_id', 'class_name', f'{best_method}_f1_mean', f'{best_method}_f1_std',
                  'raw_f1_mean', 'raw_f1_std', 'improvement']
    save_csv(csv_path, rows, fieldnames)

    # LaTeX
    tex_content = generate_perclass_f1_latex(rows, best_method)
    tex_path = os.path.join(output_dir, f'TableS6_perclass_f1_{db_key}.tex')
    save_latex(tex_path, tex_content)

    return csv_path

def generate_perclass_f1_latex(rows, best_method):
    """
    توليد محتوى LaTeX لجدول F1 لكل صنف.
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Per-Class F1 Scores — " + best_method + r" vs Raw}",
        r"\label{tab:perclass_f1}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Class & " + best_method + r" F1 & Raw F1 & Improvement \\",
        r"\midrule",
    ]

    for row in rows[:10]:  # عرض أول 10 أصناف فقط (للمختصر)
        cls = row['class_name']
        best = f"{row[f'{best_method}_f1_mean']:.3f}" if not np.isnan(row[f'{best_method}_f1_mean']) else '—'
        raw = f"{row['raw_f1_mean']:.3f}" if not np.isnan(row['raw_f1_mean']) else '—'
        imp = f"{row['improvement']:.3f}" if not np.isnan(row['improvement']) else '—'

        lines.append(f"{cls} & {best} & {raw} & {imp} \\\\")

    if len(rows) > 10:
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{4}{c}{... " + str(len(rows) - 10) + r" more classes} \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)

def generate_timing_table(db_data, db_key, output_dir):
    """
    توليد جدول أوقات التشغيل (Table S7).

    المعاملات:
        db_data (dict): نتائج قاعدة البيانات.
        db_key (str): مفتاح قاعدة البيانات.
        output_dir (str): مجلد الإخراج.

    المخرجات:
        str: مسار الملف المُنشأ.
    """
    methods = config.METHODS
    rows = []

    for method in methods:
        res = db_data.get(method)
        if res is None:
            continue

        feature_times = []
        adapt_times = []
        clf_times = []
        total_times = []

        for subj in res.get('per_subject', []):
            timing = subj.get('timing', {})
            if timing:
                feature_times.append(timing.get('feature_extraction', 0))
                adapt_times.append(timing.get('adaptation', 0))
                clf_times.append(timing.get('classification', 0))
                total_times.append(timing.get('total', 0))

        if not total_times:
            continue

        rows.append({
            'method': method,
            'method_label': config.METHOD_LABELS.get(method, method),
            'feature_extraction_mean': np.mean(feature_times) if feature_times else np.nan,
            'feature_extraction_std': np.std(feature_times, ddof=1) if len(feature_times) > 1 else np.nan,
            'adaptation_mean': np.mean(adapt_times) if adapt_times else np.nan,
            'adaptation_std': np.std(adapt_times, ddof=1) if len(adapt_times) > 1 else np.nan,
            'classification_mean': np.mean(clf_times) if clf_times else np.nan,
            'classification_std': np.std(clf_times, ddof=1) if len(clf_times) > 1 else np.nan,
            'total_mean': np.mean(total_times) if total_times else np.nan,
            'total_std': np.std(total_times, ddof=1) if len(total_times) > 1 else np.nan,
            'n_subjects': len(total_times),
        })

    if not rows:
        return None

    # CSV
    csv_path = os.path.join(output_dir, f'TableS7_timing_{db_key}.csv')
    fieldnames = ['method', 'method_label', 'feature_extraction_mean', 'feature_extraction_std',
                  'adaptation_mean', 'adaptation_std', 'classification_mean', 'classification_std',
                  'total_mean', 'total_std', 'n_subjects']
    save_csv(csv_path, rows, fieldnames)

    # LaTeX
    tex_content = generate_timing_latex(rows)
    tex_path = os.path.join(output_dir, f'TableS7_timing_{db_key}.tex')
    save_latex(tex_path, tex_content)

    return csv_path

def generate_timing_latex(rows):
    """
    توليد محتوى LaTeX لجدول أوقات التشغيل.
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Timing Comparison (seconds)}",
        r"\label{tab:timing}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & Feature Extraction & Adaptation & Classification & Total \\",
        r"\midrule",
    ]

    for row in rows:
        method = row['method_label']
        feat = f"{row['feature_extraction_mean']:.2f}" if not np.isnan(row['feature_extraction_mean']) else '—'
        adapt = f"{row['adaptation_mean']:.2f}" if not np.isnan(row['adaptation_mean']) else '—'
        clf = f"{row['classification_mean']:.2f}" if not np.isnan(row['classification_mean']) else '—'
        total = f"{row['total_mean']:.2f}" if not np.isnan(row['total_mean']) else '—'

        lines.append(f"{method} & {feat} & {adapt} & {clf} & {total} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)

def generate_cohend_table(db_data, db_key, output_dir):
    """
    توليد جدول كوهين د (Table S8).

    المعاملات:
        db_data (dict): نتائج قاعدة البيانات.
        db_key (str): مفتاح قاعدة البيانات.
        output_dir (str): مجلد الإخراج.

    المخرجات:
        str: مسار الملف المُنشأ.
    """
    methods = config.METHODS
    raw_method = 'raw'

    if raw_method not in db_data:
        return None

    # بناء مصفوفة الدقة
    acc_matrix = []
    subject_ids = [s['subject_id'] for s in db_data[raw_method].get('per_subject', [])]

    if not subject_ids:
        return None

    for method in methods:
        res = db_data.get(method)
        if res is None:
            acc_col = [np.nan] * len(subject_ids)
        else:
            acc_map = {s['subject_id']: s.get('accuracy', np.nan) for s in res.get('per_subject', [])}
            acc_col = [acc_map.get(sid, np.nan) for sid in subject_ids]
        acc_matrix.append(acc_col)

    acc_matrix = np.array(acc_matrix).T

    # إزالة الأشخاص الذين لديهم قيم مفقودة
    valid_mask = ~np.isnan(acc_matrix).any(axis=1)
    acc_matrix_clean = acc_matrix[valid_mask]

    if acc_matrix_clean.shape[0] < 2:
        return None

    raw_idx = methods.index(raw_method)
    raw_acc = acc_matrix_clean[:, raw_idx]

    rows = []
    for i, method in enumerate(methods):
        if i == raw_idx:
            continue

        method_acc = acc_matrix_clean[:, i]
        d = cohens_d(raw_acc, method_acc)
        d_interpret = interpret_cohens_d(d)

        rows.append({
            'comparison': f'{raw_method}_vs_{method}',
            'method': method,
            'cohens_d': d,
            'interpretation': d_interpret,
            'raw_mean': np.mean(raw_acc),
            'method_mean': np.mean(method_acc),
            'n_subjects': len(raw_acc),
        })

    if not rows:
        return None

    # CSV
    csv_path = os.path.join(output_dir, f'TableS8_cohend_{db_key}.csv')
    fieldnames = ['comparison', 'method', 'cohens_d', 'interpretation', 'raw_mean', 'method_mean', 'n_subjects']
    save_csv(csv_path, rows, fieldnames)

    # LaTeX
    tex_content = generate_cohend_latex(rows)
    tex_path = os.path.join(output_dir, f'TableS8_cohend_{db_key}.tex')
    save_latex(tex_path, tex_content)

    return csv_path

def generate_cohend_latex(rows):
    """
    توليد محتوى LaTeX لجدول كوهين د.
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Cohen's d Effect Size (vs Raw)}",
        r"\label{tab:cohend}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & Cohen's d & Interpretation & Mean Difference \\",
        r"\midrule",
    ]

    for row in rows:
        method = row['method']
        d = f"{row['cohens_d']:.3f}" if not np.isnan(row['cohens_d']) else '—'
        interp = row['interpretation']
        diff = f"{(row['method_mean'] - row['raw_mean']) * 100:.2f}%" if not np.isnan(row['method_mean']) else '—'

        lines.append(f"{method} & {d} & {interp} & {diff} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)

# ====================================================================
# 5. توليد الملخص النهائي (ALL_SUMMARY.txt)
# ====================================================================
def generate_all_summary(all_data, output_dir):
    """
    توليد ملف ALL_SUMMARY.txt الذي يلخص جميع النتائج.

    المعاملات:
        all_data (dict): جميع النتائج.
        output_dir (str): مجلد الإخراج.
    """
    lines = [
        "=" * 80,
        "  PAPER 2 V2: COMPREHENSIVE STATISTICAL SUMMARY",
        "  Generated: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "=" * 80,
        "",
    ]

    for db_key in ["db7", "db3", "db2"]:
        db_data = all_data.get(db_key, {})
        if not db_data:
            lines.append(f"\n  {db_key}: No data")
            continue

        db_display = config.DB_META.get(db_key, {}).get('display_name', db_key)
        lines.append(f"\n{'=' * 60}")
        lines.append(f"  {db_display}")
        lines.append(f"{'=' * 60}")

        # النتائج الرئيسية
        lines.append("\n  MAIN RESULTS:")
        lines.append(f"  {'Method':<18} {'Accuracy':<12} {'Std':<10} {'Random':<10} {'Ratio':<10}")
        lines.append("  " + "-" * 70)

        for method in config.METHODS:
            res = db_data.get(method)
            if res is None:
                continue
            summary = res.get('summary', {})
            _derive_missing_summary_fields(res)
            acc = summary.get('mean', 0) * 100
            std = summary.get('std', 0) * 100
            rand = summary.get('avg_random_baseline_pct', 0)
            ratio = summary.get('avg_ratio_vs_random', 0)
            rand = 0 if (rand is None or (isinstance(rand, float) and np.isnan(rand))) else rand
            ratio = 0 if (ratio is None or (isinstance(ratio, float) and np.isnan(ratio))) else ratio

            label = config.METHOD_LABELS.get(method, method)
            lines.append(f"  {label:<18} {acc:>6.2f}%   {std:>6.2f}%   {rand:>6.1f}%   {ratio:>6.1f}x")

        # أفضل طريقة
        best_method = max(
            [(m, r.get('summary', {}).get('mean', 0)) for m, r in db_data.items()],
            key=lambda x: x[1]
        )
        if best_method:
            lines.append(f"\n  Best Method: {config.METHOD_LABELS.get(best_method[0], best_method[0])} "
                         f"({best_method[1]*100:.2f}%)")

    lines.extend([
        "",
        "=" * 80,
        "  END OF SUMMARY",
        "=" * 80,
    ])

    # حفظ الملف
    summary_path = os.path.join(output_dir, 'ALL_SUMMARY.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"  [SAVED] {summary_path}", flush=True)

    return summary_path

# ====================================================================
# 6. واجهة سطر الأوامر (CLI)
# ====================================================================
def parse_args():
    """
    تحليل معاملات سطر الأوامر.
    """
    parser = argparse.ArgumentParser(
        description="Paper 2 V2: Comprehensive Statistical Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
أمثلة:
  # تشغيل التحليل الكامل
  python p06_run_statistical_analysis.py

  # تحديد مجلد النتائج
  python p06_run_statistical_analysis.py --results_dir ./my_results

  # تخطي تحليل الإزالة
  python p06_run_statistical_analysis.py --skip_ablation
        """
    )

    parser.add_argument("--results_dir", type=str, default=config.RESULTS_DIR,
                        help=f"مجلد النتائج (الافتراضي {config.RESULTS_DIR})")

    parser.add_argument("--ablation_dir", type=str, default=config.ABLATION_DIR,
                        help=f"مجلد الإزالة (الافتراضي {config.ABLATION_DIR})")

    parser.add_argument("--output_dir", type=str, default=config.TABLES_DIR,
                        help=f"مجلد الإخراج (الافتراضي {config.TABLES_DIR})")

    parser.add_argument("--skip_ablation", action="store_true",
                        help="تخطي تحليل الإزالة")

    return parser.parse_args()

# ====================================================================
# 7. الدالة الرئيسية (Main)
# ====================================================================
def main():
    """
    النقطة الرئيسية لتشغيل السكربت.
    """
    args = parse_args()

    # ── طباعة معلومات البداية ──
    print("=" * 70, flush=True)
    print("  PAPER 2 V2: COMPREHENSIVE STATISTICAL ANALYSIS", flush=True)
    print("  Generating All Statistical Tables and Summary", flush=True)
    print("=" * 70, flush=True)
    print(f"  Results dir:   {args.results_dir}", flush=True)
    print(f"  Ablation dir:  {args.ablation_dir}", flush=True)
    print(f"  Output dir:    {args.output_dir}", flush=True)
    print(f"  Skip ablation: {args.skip_ablation}", flush=True)
    print("=" * 70 + "\n", flush=True)

    # ── 1. تحميل النتائج الرئيسية ──
    print("  Loading main results...", flush=True)
    all_data = load_results_from_dir(args.results_dir)

    if not all_data:
        print("  [ERROR] No results found!", flush=True)
        return

    print(f"  Found data for DBs: {list(all_data.keys())}", flush=True)

    # ── 2. توليد الجداول لكل قاعدة ──
    os.makedirs(args.output_dir, exist_ok=True)

    for db_key in ["db7", "db3", "db2"]:
        db_data = all_data.get(db_key, {})
        if not db_data:
            print(f"\n  [SKIP] {db_key}: No data", flush=True)
            continue

        print(f"\n{'=' * 70}", flush=True)
        print(f"  PROCESSING: {db_key}", flush=True)
        print(f"{'=' * 70}\n", flush=True)

        # Table 2: Main Results
        print("  Generating Table 2 (Main Results)...", flush=True)
        generate_main_results_table(db_data, db_key, args.output_dir)

        # Table S1: Per Subject
        print("  Generating Table S1 (Per Subject)...", flush=True)
        generate_per_subject_table(db_data, db_key, args.output_dir)

        # Table S2: Wilcoxon
        print("  Generating Table S2 (Wilcoxon)...", flush=True)
        generate_wilcoxon_table(db_data, db_key, args.output_dir)

        # Table S3: Friedman
        print("  Generating Table S3 (Friedman)...", flush=True)
        generate_friedman_table(db_data, db_key, args.output_dir)

        # Table S4: Holm-Sidak
        print("  Generating Table S4 (Holm-Sidak)...", flush=True)
        generate_holmsidak_table(db_data, db_key, args.output_dir)

        # Table S5: Nemenyi
        print("  Generating Table S5 (Nemenyi)...", flush=True)
        generate_nemenyi_table(db_data, db_key, args.output_dir)

        # Table S6: Per-class F1
        print("  Generating Table S6 (Per-class F1)...", flush=True)
        generate_perclass_f1_table(db_data, db_key, args.output_dir)

        # Table S7: Timing
        print("  Generating Table S7 (Timing)...", flush=True)
        generate_timing_table(db_data, db_key, args.output_dir)

        # Table S8: Cohen's d
        print("  Generating Table S8 (Cohen's d)...", flush=True)
        generate_cohend_table(db_data, db_key, args.output_dir)

    # ── 3. توليد الملخص النهائي ──
    print("\n" + "=" * 70, flush=True)
    print("  GENERATING ALL_SUMMARY", flush=True)
    print("=" * 70 + "\n", flush=True)

    generate_all_summary(all_data, args.output_dir)

    # ── 4. تحليل الإزالة (اختياري) ──
    if not args.skip_ablation:
        print("\n" + "=" * 70, flush=True)
        print("  ABLATION STATISTICAL ANALYSIS", flush=True)
        print("=" * 70 + "\n", flush=True)

        # تحميل نتائج الإزالة
        ablation_data = load_ablation_results(args.ablation_dir)

        # هنا يمكن إضافة تحليل إحصائي لتجارب الإزالة
        # (مثل اختبار فريدمان على نتائج حجم النافذة)
        # هذا اختياري ويمكن إضافته حسب الحاجة

        print("  [INFO] Ablation statistical analysis not implemented in this version.")
        print("  The ablation results are available as JSON files for manual analysis.")

    # ── 5. الملخص النهائي ──
    print("\n" + "=" * 70, flush=True)
    print("  STATISTICAL ANALYSIS COMPLETED!", flush=True)
    print(f"  All tables saved to: {args.output_dir}", flush=True)
    print("=" * 70 + "\n", flush=True)

# ====================================================================
# 8. نقطة الدخول
# ====================================================================
if __name__ == "__main__":
    main()