"""
p07_generate_all_figures.py — Paper 2 V2: GOLDEN VISUALIZATION ENGINE
==================================================================
هذا السكربت هو المسؤول عن توليد جميع الأشكال البيانية النهائية للورقة.

يقرأ هذا السكربت الملفات الناتجة من الأيام السابقة:
  - ملفات JSON من p01a_run_main_loso_benchmark.py (النتائج الرئيسية)
  - ملفات JSON من p02_run_window_and_sample_ablation.py (الإزالة)
  - ملفات NPZ من p01a_run_main_loso_benchmark.py (خصائص t-SNE وأهمية النوى)
  - ملفات CSV من p04b_compute_domain_shift_metrics.py (انحراف المجال)
  - ملفات CSV من p05_analyze_kernel_importance.py (أهمية النوى)

الأشكال المُنتَجة (12 شكلاً):
  1. tsne_DB7_by_subject_before.png
  2. tsne_DB7_by_class_before.png
  3. tsne_DB7_by_subject_after_coral.png
  4. tsne_DB7_by_class_after_coral.png
  5. figure_main_results_paper2.png (5 طرق × 3 قواعد)
  6. figure_confusion_matrix_{db}.png (لكل قاعدة، أفضل طريقة)
  7. figure_kernel_importance_combined_{db}.png (لكل قاعدة)
  8. figure_domain_shift_reduction.png (من day3)
  9. figure_window_ablation.png
  10. figure_kernel_ablation.png
  11. figure_coral_reg_ablation.png
  12. figure_significance_heatmap_{db}.png (لكل قاعدة)

الاستخدام:
  python p07_generate_all_figures.py
  python p07_generate_all_figures.py --skip_tsne
  python p07_generate_all_figures.py --skip_ablation
  python p07_generate_all_figures.py --results_dir ./path/to/results
"""
import sys
import os
import json
import numpy as np
import argparse
import warnings
from datetime import datetime

# ====================================================================
# 1. إعدادات المسار والاستيرادات
# ====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import config


# ============================================================
# Movement-name helpers (added by dev_history/apply_confusion_matrix_labels_patch_already_applied.py)
# ============================================================
def get_movement_names(db_key):
    """
    Returns the (rest + movement) name list for a database key, or None if
    unavailable. DB2 and DB3 share the same Exercise B protocol and
    therefore the same name list.
    """
    if db_key == "db7":
        return getattr(config, "DB7_MOVEMENT_NAMES", None)
    if db_key in ("db2", "db3"):
        return getattr(config, "DB2_MOVEMENT_NAMES", None)
    return None


def true_movement_count(db_key, fallback_n_classes):
    """
    Returns the verified true movement count (e.g. 40 for DB7) derived from
    the configured name list, or falls back to whatever n_classes was
    passed in (e.g. from an old results JSON) if no name list is available.
    """
    names = get_movement_names(db_key)
    if names is None:
        return fallback_n_classes
    return len(names) - 1  # drop "rest"


def safe_class_labels(db_key, n_classes, max_show):
    """
    Returns a list of `max_show` tick labels for the confusion matrix axes.
    Falls back to plain numeric indices and prints a warning if the
    configured name list doesn't match n_classes in length — this is the
    guard that prevents ever silently mislabeling a class.
    """
    names = get_movement_names(db_key)
    numeric_fallback = [str(i + 1) for i in range(max_show)]

    if names is None:
        print(f"  [INFO] No movement-name list configured for '{db_key}' — "
              f"using numeric class indices.", flush=True)
        return numeric_fallback

    non_rest_names = names[1:]  # drop "rest"
    if len(non_rest_names) != n_classes or len(non_rest_names) < max_show:
        print(f"  [WARNING] Movement-name list for '{db_key}' has "
              f"{len(non_rest_names)} entries but this run has n_classes="
              f"{n_classes}. REFUSING to apply names (would risk "
              f"mislabeling classes) — falling back to numeric indices. "
              f"If n_classes looks stale (e.g. 41 for DB7), that's expected "
              f"until you rerun the LOSO pipeline with the corrected "
              f"config.py; the plot itself is still correct for the "
              f"classes actually shown.", flush=True)
        return numeric_fallback

    labels = []
    for i in range(max_show):
        full_name = non_rest_names[i]
        short = full_name if len(full_name) <= 22 else full_name[:19] + "..."
        labels.append(f"{i + 1}: {short}")
    return labels


# تجاهل التحذيرات
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# محاولة استيراد matplotlib و seaborn
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap, Normalize
    import seaborn as sns
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    warnings.warn("matplotlib/seaborn not available. Install: pip install matplotlib seaborn", ImportWarning)

# محاولة استيراد sklearn لـ t-SNE
try:
    from sklearn.manifold import TSNE
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("scikit-learn not available. t-SNE plots will be skipped.", ImportWarning)

# ====================================================================
# 2. دوال تحميل البيانات
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

def load_tsne_features(features_dir, tag):
    """
    تحميل خصائص t-SNE من ملف NPZ.

    المعاملات:
        features_dir (str): مجلد الخصائص.
        tag (str): العلامة المميزة (مثل 'db7_raw').

    المخرجات:
        dict: يحتوي على X_train, y_train, sub_train, X_test, y_test, sub_test.
    """
    npz_path = os.path.join(features_dir, f"tsne_features_{tag}.npz")
    if not os.path.exists(npz_path):
        return None

    try:
        data = np.load(npz_path, allow_pickle=True)
        return {
            'X_train': data['X_train'],
            'y_train': data['y_train'],
            'sub_train': data['sub_train'],
            'X_test': data['X_test'],
            'y_test': data['y_test'],
            'sub_test': data['sub_test'],
        }
    except Exception as e:
        print(f"  [WARN] Could not load {npz_path}: {e}", flush=True)
        return None

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
# 3. دوال الرسم البياني
# ====================================================================
def setup_plot_style():
    """
    إعداد نمط الرسوم البيانية.
    """
    if not HAS_MPL:
        return

    plt.rcParams.update({
        'font.family': config.FONT_FAMILY,
        'font.size': config.FONT_SIZE,
        'figure.dpi': config.FIGURE_DPI,
        'savefig.dpi': config.FIGURE_DPI,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.unicode_minus': False,
        'legend.framealpha': 0.9,
        'legend.edgecolor': 'white',
    })

def plot_tsne(features_dir, fig_dir, skip_tsne=False):
    """
    رسم رسوم t-SNE (4 أشكال).

    المعاملات:
        features_dir (str): مجلد الخصائص.
        fig_dir (str): مجلد الأشكال.
        skip_tsne (bool): تخطي t-SNE لتوفير الوقت.
    """
    if skip_tsne or not HAS_MPL or not HAS_SKLEARN:
        print("  [SKIP] t-SNE plots (disabled or dependencies missing)", flush=True)
        return

    setup_plot_style()
    os.makedirs(fig_dir, exist_ok=True)

    # تحميل بيانات t-SNE لـ DB7
    tags = [
        ('db7_raw', 'Raw PPV Features'),
        ('db7_coral', 'After CORAL Adaptation')
    ]

    for tag, title_suffix in tags:
        data = load_tsne_features(features_dir, tag)
        if data is None:
            print(f"  [SKIP] t-SNE data for {tag} not found", flush=True)
            continue

        # دمج بيانات التدريب والاختبار
        X = np.vstack([data['X_train'], data['X_test']])
        y = np.concatenate([data['y_train'], data['y_test']])
        sub = np.concatenate([data['sub_train'], data['sub_test']])

        # إذا كانت البيانات كبيرة جداً، نأخذ عينة عشوائية
        max_samples = 5000
        if X.shape[0] > max_samples:
            rng = np.random.RandomState(config.RANDOM_SEED)
            idx = rng.choice(X.shape[0], max_samples, replace=False)
            X = X[idx]
            y = y[idx]
            sub = sub[idx]

        print(f"  t-SNE {tag}: {X.shape[0]} points, fitting...", flush=True)

        try:
            # حساب t-SNE
            tsne = TSNE(
                n_components=2,
                random_state=config.RANDOM_SEED,
                perplexity=30,
                learning_rate=200,
                n_iter=1000,
                init='pca'
            )
            embedding = tsne.fit_transform(X)

            # ── شكل 1: حسب الشخص (Subject ID) ──
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(
                embedding[:, 0], embedding[:, 1],
                c=sub, cmap='tab20', s=8, alpha=0.7
            )
            cbar = plt.colorbar(scatter, ax=ax, label='Subject ID', shrink=0.8)
            ax.set_title(f"PPV Features — {title_suffix}\nColored by Subject ID",
                         fontweight='bold', fontsize=13)
            ax.set_xlabel('t-SNE Dimension 1')
            ax.set_ylabel('t-SNE Dimension 2')
            ax.grid(alpha=0.2)

            # حفظ الشكل
            suffix = 'after_coral' if 'coral' in tag else 'before'
            save_path = os.path.join(fig_dir, f"tsne_DB7_by_subject_{suffix}.png")
            plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            print(f"    [SAVED] {save_path}", flush=True)

            # ── شكل 2: حسب الصنف (Class Label) ──
            fig, ax = plt.subplots(figsize=(10, 8))
            n_classes = len(np.unique(y))
            cmap = 'tab20' if n_classes <= 20 else 'tab20b' if n_classes <= 40 else 'viridis'
            scatter = ax.scatter(
                embedding[:, 0], embedding[:, 1],
                c=y, cmap=cmap, s=8, alpha=0.7
            )
            cbar = plt.colorbar(scatter, ax=ax, label='Gesture Class', shrink=0.8)
            ax.set_title(f"PPV Features — {title_suffix}\nColored by Gesture Class",
                         fontweight='bold', fontsize=13)
            ax.set_xlabel('t-SNE Dimension 1')
            ax.set_ylabel('t-SNE Dimension 2')
            ax.grid(alpha=0.2)

            # حفظ الشكل
            save_path = os.path.join(fig_dir, f"tsne_DB7_by_class_{suffix}.png")
            plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            print(f"    [SAVED] {save_path}", flush=True)

        except Exception as e:
            print(f"    [ERROR] t-SNE failed for {tag}: {e}", flush=True)

def plot_main_results(all_data, fig_dir):
    """
    رسم النتائج الرئيسية (5 طرق × 3 قواعد).

    المعاملات:
        all_data (dict): جميع النتائج.
        fig_dir (str): مجلد الأشكال.
    """
    if not HAS_MPL:
        return

    setup_plot_style()
    os.makedirs(fig_dir, exist_ok=True)

    methods = config.METHODS
    databases = ['db7', 'db3', 'db2']
    n_db = len(databases)
    n_m = len(methods)

    # جمع البيانات
    means = np.full((n_db, n_m), np.nan)
    stds = np.full((n_db, n_m), np.nan)
    ci_lower = np.full((n_db, n_m), np.nan)
    ci_upper = np.full((n_db, n_m), np.nan)

    for i, db_key in enumerate(databases):
        db_data = all_data.get(db_key, {})
        for j, method in enumerate(methods):
            res = db_data.get(method)
            if res is None:
                continue
            summary = res.get('summary', {})
            means[i, j] = summary.get('mean', np.nan) * 100  # كنسبة مئوية
            stds[i, j] = summary.get('std', np.nan) * 100
            ci_lower[i, j] = summary.get('ci95_lower', np.nan) * 100 if summary.get('ci95_lower') is not None else means[i, j] - 1.96 * stds[i, j]
            ci_upper[i, j] = summary.get('ci95_upper', np.nan) * 100 if summary.get('ci95_upper') is not None else means[i, j] + 1.96 * stds[i, j]

    # إنشاء الشكل
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(n_db)
    width = 0.15

    for j, method in enumerate(methods):
        offset = (j - (n_m - 1) / 2) * width
        color = config.METHOD_COLORS.get(method, '#555555')
        label = config.METHOD_LABELS.get(method, method)

        # حساب أشرطة الخطأ
        err_lo = means[:, j] - ci_lower[:, j]
        err_hi = ci_upper[:, j] - means[:, j]
        # matplotlib يتوقع yerr بشكل (2, N) للأشرطة غير المتماثلة
        err_lo = np.nan_to_num(err_lo, nan=0.0)
        err_hi = np.nan_to_num(err_hi, nan=0.0)
        yerr = np.vstack([err_lo, err_hi])

        bars = ax.bar(
            x + offset, means[:, j], width,
            yerr=yerr,
            label=label,
            color=color,
            alpha=0.85,
            capsize=4,
            edgecolor='white',
            linewidth=0.5,
            error_kw={'elinewidth': 1.5, 'capthick': 1.5}
        )

        # إضافة قيم الدقة فوق الأشرطة
        for bar, val in zip(bars, means[:, j]):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.1f}",
                    ha='center', va='bottom',
                    fontsize=7, fontweight='bold'
                )

    # تنسيق المحاور
    ax.set_xticks(x)
    ax.set_xticklabels([
        config.DB_META.get(db, {}).get('display_name', db)
        for db in databases
    ], fontsize=10)

    ax.set_ylabel('LOSO Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_ylim(bottom=max(0, np.nanmin(means) - 10), top=min(100, np.nanmax(means) + 5))

    ax.legend(loc='upper right', fontsize=9, ncol=2, framealpha=0.95)
    ax.set_title('MiniROCKET Cross-Subject sEMG Classification\n'
                 'Effect of Domain Adaptation Methods',
                 fontsize=14, fontweight='bold')

    # إزالة الحدود العلوية واليمنى
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(fig_dir, 'figure_main_results_paper2.png')
    plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [SAVED] {save_path}", flush=True)

def plot_confusion_matrices(all_data, fig_dir):
    """
    رسم مصفوفات الارتباك لأفضل طريقة في كل قاعدة.

    المعاملات:
        all_data (dict): جميع النتائج.
        fig_dir (str): مجلد الأشكال.
    """
    if not HAS_MPL:
        return

    setup_plot_style()
    os.makedirs(fig_dir, exist_ok=True)

    for db_key in ['db7', 'db3', 'db2']:
        db_data = all_data.get(db_key, {})
        if not db_data:
            continue

        # تحديد أفضل طريقة (أعلى متوسط دقة)
        best_method = None
        best_acc = -1
        for method, res in db_data.items():
            acc = res.get('summary', {}).get('mean', 0)
            if acc > best_acc:
                best_acc = acc
                best_method = method

        if best_method is None:
            continue

        res = db_data[best_method]
        # FIX-P07-COMPAT: نفس فجوة مخطط p06 — النتائج الجديدة (V3) لا
        # تضع n_classes على المستوى الأعلى، فكان هذا يُرجع 0 دائماً
        # ويتخطى رسم مصفوفة الارتباك بصمت لكل قاعدة بيانات جديدة.
        n_classes = res.get('n_classes', 0)
        if not n_classes:
            for s in res.get('per_subject', []):
                fpc = s.get('f1_per_class')
                if fpc:
                    n_classes = len(fpc)
                    break
                cm0 = s.get('confusion_matrix')
                if cm0:
                    n_classes = len(cm0)
                    break

        if n_classes == 0:
            continue

        # تجميع مصفوفات الارتباك من جميع الأشخاص
        agg_cm = np.zeros((n_classes, n_classes), dtype=float)
        n_subjects = 0

        for subj in res.get('per_subject', []):
            cm = subj.get('confusion_matrix')
            if cm is None:
                continue
            cm = np.array(cm)
            if cm.shape == (n_classes, n_classes):
                # تطبيع كل صف (نسبة لكل صنف حقيقي)
                row_sums = cm.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1
                agg_cm += cm / row_sums
                n_subjects += 1

        if n_subjects == 0:
            continue

        agg_cm /= n_subjects

        # تحديد عدد الأصناف المعروضة (حد أقصى 20 للتشغيل السريع)
        max_show = min(20, n_classes)
        cm_show = agg_cm[:max_show, :max_show]

        # رسم مصفوفة الارتباك
        fig, ax = plt.subplots(figsize=(12, 10))

        im = ax.imshow(cm_show, cmap='Blues', vmin=0, vmax=1, interpolation='nearest')
        cbar = plt.colorbar(im, ax=ax, label='Normalized Frequency', shrink=0.8)

        # Movement-name tick labels, with a hard safety fallback to numeric
        # indices if the configured name list doesn't match n_classes exactly.
        true_n = true_movement_count(db_key, n_classes)
        tick_labels = safe_class_labels(db_key, true_n, max_show)
        ax.set_xticks(range(max_show))
        ax.set_yticks(range(max_show))
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=7, ha='center')
        ax.set_yticklabels(tick_labels, fontsize=7)

        ax.set_xlabel('Predicted Class', fontsize=12, labelpad=10)
        ax.set_ylabel('True Class', fontsize=12)

        db_display = config.DB_META.get(db_key, {}).get('display_name', db_key)
        method_label = config.METHOD_LABELS.get(best_method, best_method)
        ax.set_title(f'Aggregated Confusion Matrix — {db_display}\n'
                     f'Best Method: {method_label} ({best_acc*100:.1f}%)',
                     fontweight='bold', fontsize=12)

        if max_show < true_n:
            ax.text(
                0.5, -0.08,
                f'(Showing first {max_show} of {true_n} classes)',
                transform=ax.transAxes, ha='center', va='top',
                fontsize=9, style='italic'
            )

        # إزالة الحدود
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        save_path = os.path.join(fig_dir, f'figure_confusion_matrix_{db_key}.png')
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  [SAVED] {save_path}", flush=True)

def plot_significance_heatmaps(all_data, fig_dir):
    """
    رسم خرائط حرارة الدلالة الإحصائية (قيم p لاختبار ويلكوكسون).

    المعاملات:
        all_data (dict): جميع النتائج.
        fig_dir (str): مجلد الأشكال.
    """
    if not HAS_MPL:
        return

    try:
        from scipy import stats as sp_stats
    except ImportError:
        print("  [WARN] scipy not available. Skipping significance heatmaps.", flush=True)
        return

    setup_plot_style()
    os.makedirs(fig_dir, exist_ok=True)

    methods = config.METHODS

    for db_key in ['db7', 'db3', 'db2']:
        db_data = all_data.get(db_key, {})
        if not db_data:
            continue

        # التحقق من وجود جميع الطرق
        available_methods = [m for m in methods if m in db_data]
        if len(available_methods) < 2:
            continue

        n_m = len(available_methods)

        # بناء مصفوفة الدقة
        subject_ids = []
        for method in available_methods:
            res = db_data.get(method)
            if res is not None:
                subject_ids = [s['subject_id'] for s in res.get('per_subject', [])]
                break

        if not subject_ids:
            continue

        acc_matrix = []
        for method in available_methods:
            res = db_data.get(method)
            if res is None:
                acc_col = [np.nan] * len(subject_ids)
            else:
                acc_map = {s['subject_id']: s.get('accuracy', np.nan) for s in res.get('per_subject', [])}
                acc_col = [acc_map.get(sid, np.nan) for sid in subject_ids]
            acc_matrix.append(acc_col)

        acc_matrix = np.array(acc_matrix).T
        valid_mask = ~np.isnan(acc_matrix).any(axis=1)
        acc_matrix_clean = acc_matrix[valid_mask]

        if acc_matrix_clean.shape[0] < 3:
            continue

        # حساب قيم p لكل زوج
        p_matrix = np.ones((n_m, n_m))
        for i in range(n_m):
            for j in range(n_m):
                if i >= j:
                    continue
                try:
                    _, p = sp_stats.wilcoxon(acc_matrix_clean[:, i], acc_matrix_clean[:, j])
                    p_matrix[i, j] = p
                    p_matrix[j, i] = p
                except Exception:
                    p_matrix[i, j] = 1.0
                    p_matrix[j, i] = 1.0

        # رسم خريطة الحرارة
        fig, ax = plt.subplots(figsize=(8, 6))

        # استخدام مقياس لوغاريتمي للقيم الصغيرة
        # نعرض القيم مباشرة مع ترميز اللون
        im = ax.imshow(p_matrix, cmap='RdYlGn_r', vmin=0, vmax=0.1, interpolation='nearest')
        cbar = plt.colorbar(im, ax=ax, label='Wilcoxon p-value')
        cbar.ax.tick_params(labelsize=9)

        ax.set_xticks(range(n_m))
        ax.set_yticks(range(n_m))
        ax.set_xticklabels([config.METHOD_LABELS.get(m, m) for m in available_methods],
                           rotation=30, ha='right', fontsize=9)
        ax.set_yticklabels([config.METHOD_LABELS.get(m, m) for m in available_methods],
                           fontsize=9)

        # إضافة القيم داخل الخلايا
        for i in range(n_m):
            for j in range(n_m):
                if i == j:
                    ax.text(j, i, '1.000', ha='center', va='center', fontsize=8, color='black')
                else:
                    p_val = p_matrix[i, j]
                    sig_stars = ''
                    if p_val < 0.001:
                        sig_stars = '***'
                    elif p_val < 0.01:
                        sig_stars = '**'
                    elif p_val < 0.05:
                        sig_stars = '*'
                    text_color = 'white' if p_val < 0.05 else 'black'
                    ax.text(j, i, f'{p_val:.3f}{sig_stars}',
                            ha='center', va='center', fontsize=7,
                            color=text_color)

        db_display = config.DB_META.get(db_key, {}).get('display_name', db_key)
        ax.set_title(f'Pairwise Significance — {db_display}\n'
                     f'(* p<0.05, ** p<0.01, *** p<0.001)',
                     fontweight='bold', fontsize=11)

        plt.tight_layout()

        save_path = os.path.join(fig_dir, f'figure_significance_heatmap_{db_key}.png')
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  [SAVED] {save_path}", flush=True)

def plot_ablation_summary(ablation_dir, fig_dir):
    """
    رسم أشكال الإزالة (حجم النافذة، عدد النوى، معامل CORAL، تجميع الأصناف).

    المعاملات:
        ablation_dir (str): مجلد الإزالة.
        fig_dir (str): مجلد الأشكال.
    """
    if not HAS_MPL:
        return

    setup_plot_style()
    os.makedirs(fig_dir, exist_ok=True)

    ablation_data = load_ablation_results(ablation_dir)

    # ── 1. حجم النافذة ──
    window_data = ablation_data['window']
    if window_data:
        window_sizes = []
        means = []
        stds = []

        for fname, res in sorted(window_data.items()):
            import re
            match = re.search(r'window(\d+)ms', fname)
            if match:
                wms = int(match.group(1))
                window_sizes.append(wms)
                summary = res.get('summary', {})
                means.append(summary.get('mean', 0) * 100)
                stds.append(summary.get('std', 0) * 100)

        if window_sizes:
            sorted_idx = np.argsort(window_sizes)
            window_sizes = np.array(window_sizes)[sorted_idx]
            means = np.array(means)[sorted_idx]
            stds = np.array(stds)[sorted_idx]

            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ['#BBDEFB', '#64B5F6', '#1E88E5', '#0D47A1']
            bars = ax.bar(range(len(window_sizes)), means, yerr=stds,
                          color=colors, alpha=0.85, capsize=5,
                          error_kw={'elinewidth': 1.5, 'capthick': 1.5})
            ax.set_xticks(range(len(window_sizes)))
            ax.set_xticklabels([f'{w} ms' for w in window_sizes], fontsize=10)
            ax.set_ylabel('LOSO Accuracy (%)', fontsize=12, fontweight='bold')
            ax.set_title('Ablation: Window Size (DB7, CORAL)', fontweight='bold', fontsize=13)
            ax.set_ylim(bottom=max(0, min(means) - 5), top=min(100, max(means) + 5))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', alpha=0.3)

            # إضافة القيم
            for bar, val in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

            plt.tight_layout()
            save_path = os.path.join(fig_dir, 'figure_ablation_window_size.png')
            plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            print(f"  [SAVED] {save_path}", flush=True)

    # ── 2. عدد النوى ──
    kernel_data = ablation_data['kernel']
    if kernel_data:
        kernel_counts = []
        means = []
        stds = []

        for fname, res in sorted(kernel_data.items()):
            import re
            match = re.search(r'kernels(\d+)', fname)
            if match:
                nk = int(match.group(1))
                kernel_counts.append(nk)
                summary = res.get('summary', {})
                means.append(summary.get('mean', 0) * 100)
                stds.append(summary.get('std', 0) * 100)

        if kernel_counts:
            sorted_idx = np.argsort(kernel_counts)
            kernel_counts = np.array(kernel_counts)[sorted_idx]
            means = np.array(means)[sorted_idx]
            stds = np.array(stds)[sorted_idx]

            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ['#C8E6C9', '#66BB6A', '#2E7D32']
            bars = ax.bar(range(len(kernel_counts)), means, yerr=stds,
                          color=colors, alpha=0.85, capsize=5,
                          error_kw={'elinewidth': 1.5, 'capthick': 1.5})
            ax.set_xticks(range(len(kernel_counts)))
            ax.set_xticklabels([f'{k:,}' for k in kernel_counts], fontsize=10)
            ax.set_ylabel('LOSO Accuracy (%)', fontsize=12, fontweight='bold')
            ax.set_title('Ablation: Number of Kernels (DB7, CORAL)', fontweight='bold', fontsize=13)
            ax.set_ylim(bottom=max(0, min(means) - 5), top=min(100, max(means) + 5))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', alpha=0.3)

            for bar, val in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

            plt.tight_layout()
            save_path = os.path.join(fig_dir, 'figure_ablation_kernel_count.png')
            plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            print(f"  [SAVED] {save_path}", flush=True)

    # ── 3. معامل تنظيم CORAL ──
    coral_reg_data = ablation_data['coral_reg']
    if coral_reg_data:
        reg_values = []
        means = []
        stds = []

        # إضافة الافتراضي (1e-3) من مجلد النتائج الرئيسي
        main_results_dir = os.path.join(os.path.dirname(ablation_dir), 'results')
        default_path = os.path.join(main_results_dir, 'MiniROCKET_db7_coral_results.json')
        if os.path.exists(default_path):
            try:
                with open(default_path, 'r', encoding='utf-8') as f:
                    default_res = json.load(f)
                coral_reg_data['reg1e-3'] = default_res
            except Exception:
                pass

        for fname, res in coral_reg_data.items():
            import re
            match = re.search(r'reg([\de\-]+)', fname)
            if match:
                reg_str = match.group(1).replace('e-', 'e-')
                try:
                    reg_val = float(reg_str)
                    reg_values.append(reg_val)
                    summary = res.get('summary', {})
                    means.append(summary.get('mean', 0) * 100)
                    stds.append(summary.get('std', 0) * 100)
                except ValueError:
                    continue

        if reg_values:
            sorted_idx = np.argsort(reg_values)
            reg_values = np.array(reg_values)[sorted_idx]
            means = np.array(means)[sorted_idx]
            stds = np.array(stds)[sorted_idx]

            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ['#FFE0B2', '#FFB74D', '#F57C00']
            bars = ax.bar(range(len(reg_values)), means, yerr=stds,
                          color=colors, alpha=0.85, capsize=5,
                          error_kw={'elinewidth': 1.5, 'capthick': 1.5})
            ax.set_xticks(range(len(reg_values)))
            ax.set_xticklabels([f'{r:.0e}' for r in reg_values], fontsize=10)
            ax.set_xlabel('CORAL Regularization', fontsize=12, fontweight='bold')
            ax.set_ylabel('LOSO Accuracy (%)', fontsize=12, fontweight='bold')
            ax.set_title('Ablation: CORAL Regularization (DB7)', fontweight='bold', fontsize=13)
            ax.set_ylim(bottom=max(0, min(means) - 5), top=min(100, max(means) + 5))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', alpha=0.3)

            for bar, val in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

            plt.tight_layout()
            save_path = os.path.join(fig_dir, 'figure_ablation_coral_reg.png')
            plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            print(f"  [SAVED] {save_path}", flush=True)

    # ── 4. تجميع الأصناف ──
    grouping_data = ablation_data['grouping']
    if grouping_data:
        db_names = []
        n_classes = []
        means = []
        stds = []

        # إضافة النتائج غير المجمعة للمقارنة
        main_results_dir = os.path.join(os.path.dirname(ablation_dir), 'results')
        for db_key in ['db7', 'db3']:
            default_path = os.path.join(main_results_dir, f'MiniROCKET_{db_key}_coral_results.json')
            if os.path.exists(default_path):
                try:
                    with open(default_path, 'r', encoding='utf-8') as f:
                        res = json.load(f)
                    db_names.append(f'{db_key}_ungrouped')
                    n_classes.append(res.get('n_classes', '?'))
                    summary = res.get('summary', {})
                    means.append(summary.get('mean', 0) * 100)
                    stds.append(summary.get('std', 0) * 100)
                except Exception:
                    pass

        for fname, res in grouping_data.items():
            db_key = 'db7' if 'db7' in fname else 'db3' if 'db3' in fname else 'unknown'
            db_names.append(f'{db_key}_grouped')
            n_classes.append(res.get('n_classes', '?'))
            summary = res.get('summary', {})
            means.append(summary.get('mean', 0) * 100)
            stds.append(summary.get('std', 0) * 100)

        if db_names:
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2']
            bars = ax.bar(range(len(db_names)), means, yerr=stds,
                          color=colors[:len(db_names)], alpha=0.85, capsize=5,
                          error_kw={'elinewidth': 1.5, 'capthick': 1.5})
            ax.set_xticks(range(len(db_names)))
            labels = [f'{n.replace("_"," ")}\n({c} cls)' for n, c in zip(db_names, n_classes)]
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_ylabel('LOSO Accuracy (%)', fontsize=12, fontweight='bold')
            ax.set_title('Ablation: Class Grouping (CORAL)', fontweight='bold', fontsize=13)
            ax.set_ylim(bottom=max(0, min(means) - 5), top=min(100, max(means) + 5))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', alpha=0.3)

            for bar, val in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

            plt.tight_layout()
            save_path = os.path.join(fig_dir, 'figure_ablation_grouping.png')
            plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            print(f"  [SAVED] {save_path}", flush=True)

def plot_kernel_importance_combined(features_dir, fig_dir):
    """
    رسم أهمية النوى حسب التمدد (مدمج raw vs coral) لكل قاعدة.

    المعاملات:
        features_dir (str): مجلد الخصائص.
        fig_dir (str): مجلد الأشكال.
    """
    if not HAS_MPL:
        return

    setup_plot_style()
    os.makedirs(fig_dir, exist_ok=True)

    for db_key in ['db7', 'db3', 'db2']:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax_idx, method in enumerate(['raw', 'coral']):
            ax = axes[ax_idx]

            # تحميل بيانات الأهمية
            npz_path = os.path.join(features_dir, f"kernel_imp_{db_key}_{method}.npz")
            if not os.path.exists(npz_path):
                ax.text(0.5, 0.5, f'No data for {method}', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12)
                ax.set_title(config.METHOD_LABELS.get(method, method), fontsize=11)
                continue

            try:
                data = np.load(npz_path, allow_pickle=True)
                imp = data['importance_matrix']  # (n_folds, n_kernels)
                dils = data['dilations']

                unique_d = np.sort(np.unique(dils))
                means = [float(imp[:, dils == d].mean()) for d in unique_d]
                stds = [float(imp[:, dils == d].std()) for d in unique_d]

                color = config.METHOD_COLORS.get(method, '#555555')
                ax.bar(range(len(unique_d)), means, yerr=stds,
                       color=color, alpha=0.85, capsize=3,
                       error_kw={'elinewidth': 1, 'capthick': 1})
                ax.set_xticks(range(len(unique_d)))
                ax.set_xticklabels([str(int(d)) for d in unique_d], fontsize=8)
                ax.set_xlabel('Dilation', fontsize=10)
                ax.set_ylabel('Mean Kernel Importance', fontsize=10)
                ax.set_title(f"{config.METHOD_LABELS.get(method, method)}", fontsize=11)
                ax.grid(axis='y', alpha=0.3)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            except Exception as e:
                ax.text(0.5, 0.5, f'Error: {e}', ha='center', va='center',
                        transform=ax.transAxes, fontsize=10)
                ax.set_title(config.METHOD_LABELS.get(method, method), fontsize=11)

        db_display = config.DB_META.get(db_key, {}).get('display_name', db_key)
        fig.suptitle(f'Kernel Importance by Dilation — {db_display}',
                     fontsize=12, fontweight='bold')

        plt.tight_layout()
        save_path = os.path.join(fig_dir, f'figure_kernel_importance_combined_{db_key}.png')
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  [SAVED] {save_path}", flush=True)

# ====================================================================
# 4. واجهة سطر الأوامر (CLI)
# ====================================================================
def parse_args():
    """
    تحليل معاملات سطر الأوامر.
    """
    parser = argparse.ArgumentParser(
        description="Paper 2 V2: Golden Visualization Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
أمثلة:
  # تشغيل جميع الأشكال
  python p07_generate_all_figures.py

  # تخطي t-SNE (لتوفير الوقت)
  python p07_generate_all_figures.py --skip_tsne

  # تخطي أشكال الإزالة
  python p07_generate_all_figures.py --skip_ablation

  # تحديد مجلد النتائج
  python p07_generate_all_figures.py --results_dir ./my_results
        """
    )

    parser.add_argument("--results_dir", type=str, default=config.RESULTS_DIR,
                        help=f"مجلد النتائج (الافتراضي {config.RESULTS_DIR})")

    parser.add_argument("--features_dir", type=str, default=config.FEATURES_DIR,
                        help=f"مجلد الخصائص (الافتراضي {config.FEATURES_DIR})")

    parser.add_argument("--ablation_dir", type=str, default=config.ABLATION_DIR,
                        help=f"مجلد الإزالة (الافتراضي {config.ABLATION_DIR})")

    parser.add_argument("--figures_dir", type=str, default=config.FIGURES_DIR,
                        help=f"مجلد الأشكال (الافتراضي {config.FIGURES_DIR})")

    parser.add_argument("--skip_tsne", action="store_true",
                        help="تخطي رسوم t-SNE (لتوفير الوقت)")

    parser.add_argument("--skip_ablation", action="store_true",
                        help="تخطي أشكال الإزالة")

    return parser.parse_args()

# ====================================================================
# 5. الدالة الرئيسية (Main)
# ====================================================================
def main():
    """
    النقطة الرئيسية لتشغيل السكربت.
    """
    args = parse_args()

    # ── طباعة معلومات البداية ──
    print("=" * 70, flush=True)
    print("  PAPER 2 V2: GOLDEN VISUALIZATION ENGINE", flush=True)
    print("  Generating All Publication-Quality Figures", flush=True)
    print("=" * 70, flush=True)
    print(f"  Results dir:    {args.results_dir}", flush=True)
    print(f"  Features dir:   {args.features_dir}", flush=True)
    print(f"  Ablation dir:   {args.ablation_dir}", flush=True)
    print(f"  Figures dir:    {args.figures_dir}", flush=True)
    print(f"  Skip t-SNE:     {args.skip_tsne}", flush=True)
    print(f"  Skip ablation:  {args.skip_ablation}", flush=True)
    print("=" * 70 + "\n", flush=True)

    # ── التحقق من matplotlib ──
    if not HAS_MPL:
        print("  [ERROR] matplotlib not available. Install: pip install matplotlib seaborn", flush=True)
        return

    # ── إنشاء مجلد الأشكال ──
    os.makedirs(args.figures_dir, exist_ok=True)

    # ── 1. تحميل النتائج الرئيسية ──
    print("  Loading main results...", flush=True)
    all_data = load_results_from_dir(args.results_dir)

    if not all_data:
        print("  [WARN] No results found. Some figures will be skipped.", flush=True)

    # ── 2. رسوم t-SNE ──
    print("\n" + "=" * 70, flush=True)
    print("  1-4. t-SNE PLOTS", flush=True)
    print("=" * 70 + "\n", flush=True)

    plot_tsne(args.features_dir, args.figures_dir, args.skip_tsne)

    # ── 3. النتائج الرئيسية ──
    print("\n" + "=" * 70, flush=True)
    print("  5. MAIN RESULTS BAR CHART", flush=True)
    print("=" * 70 + "\n", flush=True)

    if all_data:
        plot_main_results(all_data, args.figures_dir)
    else:
        print("  [SKIP] Main results (no data)", flush=True)

    # ── 4. مصفوفات الارتباك ──
    print("\n" + "=" * 70, flush=True)
    print("  6. CONFUSION MATRICES", flush=True)
    print("=" * 70 + "\n", flush=True)

    if all_data:
        plot_confusion_matrices(all_data, args.figures_dir)
    else:
        print("  [SKIP] Confusion matrices (no data)", flush=True)

    # ── 5. خرائط حرارة الدلالة الإحصائية ──
    print("\n" + "=" * 70, flush=True)
    print("  7. SIGNIFICANCE HEATMAPS", flush=True)
    print("=" * 70 + "\n", flush=True)

    if all_data:
        plot_significance_heatmaps(all_data, args.figures_dir)
    else:
        print("  [SKIP] Significance heatmaps (no data)", flush=True)

    # ── 6. أهمية النوى ──
    print("\n" + "=" * 70, flush=True)
    print("  8. KERNEL IMPORTANCE", flush=True)
    print("=" * 70 + "\n", flush=True)

    plot_kernel_importance_combined(args.features_dir, args.figures_dir)

    # ── 7. أشكال الإزالة ──
    print("\n" + "=" * 70, flush=True)
    print("  9. ABLATION PLOTS", flush=True)
    print("=" * 70 + "\n", flush=True)

    if not args.skip_ablation:
        plot_ablation_summary(args.ablation_dir, args.figures_dir)
    else:
        print("  [SKIP] Ablation plots", flush=True)

    # ── 8. انخفاض انحراف المجال (من day3) ──
    print("\n" + "=" * 70, flush=True)
    print("  10. DOMAIN SHIFT REDUCTION", flush=True)
    print("=" * 70 + "\n", flush=True)

    # محاولة استيراد day3 لتشغيل الرسم البياني
    try:
        import p04b_compute_domain_shift_metrics as d3
        d3.plot_domain_shift_reduction(all_data, args.figures_dir)
    except ImportError:
        print("  [WARN] Could not import p04b_compute_domain_shift_metrics", flush=True)
    except Exception as e:
        print(f"  [WARN] Domain shift plot failed: {e}", flush=True)

    # ── الملخص النهائي ──
    print("\n" + "=" * 70, flush=True)
    print("  VISUALIZATION COMPLETED!", flush=True)
    print(f"  All figures saved to: {args.figures_dir}", flush=True)
    print("=" * 70 + "\n", flush=True)

# ====================================================================
# 6. نقطة الدخول
# ====================================================================
if __name__ == "__main__":
    main()
