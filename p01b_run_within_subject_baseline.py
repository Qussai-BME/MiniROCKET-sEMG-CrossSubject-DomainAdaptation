"""
p01b_run_within_subject_baseline.py — Paper 2 V2: Within-Subject Baseline
========================================================================
الغرض:
  توفير رقم حقيقي مُختبَر لدقة "ضمن-الشخص" (within-subject) بنفس الـ
  pipeline بالضبط (MiniROCKET + نفس حجم النافذة + نفس المصنّف)، ليكون
  مقارنة تفاح-بتفاح حقيقية مقابل نتائج LOSO في Table 2 — بدل الاعتماد
  فقط على أرقام تقريبية من الأدبيات.

المنطق:
  لكل شخص على حدة: نقسّم نوافذه هو فقط إلى تدريب/اختبار (80/20، طبقي
  عبر الأصناف)، نُدرّب ونختبر عليه فقط. لا يوجد تكيّف مجالي هنا إطلاقاً
  (raw فقط) لأنه لا يوجد انزياح بين التدريب والاختبار أصلاً — نفس الشخص.

  هذا أرخص حسابياً بكثير من LOSO: كل fold يحتوي بيانات شخص واحد فقط
  (~5000 نافذة كحد أقصى) بدل تجميع عشرات الأشخاص.

الاستخدام:
  python p01b_run_within_subject_baseline.py --db db7
  python p01b_run_within_subject_baseline.py --all
  python p01b_run_within_subject_baseline.py --db db7 --n_seeds 3   # لفحص
      حساسية تقسيم التدريب/الاختبار (تنويع إضافي اختياري، انظر ملاحظة
      أسفل الملف حول الفرق بين هذا وحساسية بذرة نوى MiniROCKET نفسها)
"""
import sys, os, json, time, argparse, warnings
from datetime import datetime
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import config
from data_loader import load_ninapro_stream, validate_data_path, get_n_classes
from minirocket import MiniRocketPipeline
from statistical_utils import descriptive_stats, random_baseline_from_labels, ratio_vs_random

warnings.filterwarnings('ignore')


def stratified_within_subject_split(X, y, test_size=0.2, random_state=42):
    """تقسيم طبقي 80/20 لنوافذ شخص واحد، صنفاً بصنف لضمان تمثيل كل الأصناف
    في كلا الطرفين حتى مع أصناف قليلة العدد."""
    rng = np.random.RandomState(random_state)
    train_idx, test_idx = [], []
    for c in np.unique(y):
        cls_idx = np.where(y == c)[0]
        rng.shuffle(cls_idx)
        n_test = max(1, int(round(len(cls_idx) * test_size)))
        test_idx.extend(cls_idx[:n_test])
        train_idx.extend(cls_idx[n_test:])
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def run_within_subject_fold(X, y, num_kernels, random_state=42, split_seed=42):
    X = X.astype(np.float32)
    train_idx, test_idx = stratified_within_subject_split(X, y, test_size=0.2, random_state=split_seed)
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    rocket = MiniRocketPipeline(num_kernels=num_kernels, random_state=random_state)
    X_train_ppv = rocket.rocket.fit_transform(X_train)
    X_test_ppv = rocket.rocket.transform(X_test)

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeClassifierCV
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_ppv)
    X_test_s = scaler.transform(X_test_ppv)

    clf = RidgeClassifierCV(alphas=np.logspace(-3, 6, 25), scoring='accuracy', cv=None)
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)
    accuracy = float(np.mean(y_pred == y_test))

    n_classes_present = len(np.unique(y))
    random_baseline_pct = random_baseline_from_labels(y_test) if 'random_baseline_from_labels' in dir() else 100.0 / n_classes_present
    ratio = ratio_vs_random(accuracy, random_baseline_pct) if 'ratio_vs_random' in dir() else (accuracy * 100.0 / random_baseline_pct if random_baseline_pct else np.nan)

    return {
        "accuracy": accuracy,
        "n_train_samples": int(len(y_train)),
        "n_test_samples": int(len(y_test)),
        "n_classes": int(n_classes_present),
        "random_baseline_pct": float(random_baseline_pct),
        "ratio_vs_random": float(ratio),
    }


def run_within_subject_db(db_key, num_kernels=5000, window_ms=None, overlap=None,
                            per_subject_limit=5000, n_seeds=1, output_dir=None):
    window_ms = window_ms or config.DEFAULT_WINDOW_MS
    overlap = overlap or config.DEFAULT_OVERLAP
    output_dir = output_dir or config.RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    validate_data_path(db_key, config)
    n_classes = get_n_classes(db_key, config)

    all_subject_results = []
    stream = load_ninapro_stream(db_key, config, window_ms, overlap)

    for subj in stream:
        sid = subj["subject_id"]
        X, y = subj["X"], subj["y"]

        # نفس منطق التصغير المستخدم في day2 لضبط حجم العينة الأقصى للشخص
        if len(y) > per_subject_limit:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(y), per_subject_limit, replace=False)
            X, y = X[idx], y[idx]

        seed_accs = []
        for seed in range(n_seeds):
            t0 = time.time()
            fold = run_within_subject_fold(X, y, num_kernels,
                                            random_state=config.RANDOM_SEED,
                                            split_seed=42 + seed)
            fold["time_sec"] = round(time.time() - t0, 2)
            fold["split_seed"] = 42 + seed
            seed_accs.append(fold["accuracy"])
            print(f"  [{db_key}] Subject {sid}, split_seed={42+seed}: "
                  f"acc={fold['accuracy']*100:.2f}% ({fold['time_sec']:.1f}s)", flush=True)

        all_subject_results.append({
            "subject_id": sid,
            "accuracy_mean": float(np.mean(seed_accs)),
            "accuracy_std": float(np.std(seed_accs)) if n_seeds > 1 else 0.0,
            "n_splits": n_seeds,
            "per_split_accuracy": seed_accs,
        })

    accs = [r["accuracy_mean"] for r in all_subject_results]
    desc = descriptive_stats(accs) if accs else {}

    result = {
        "database": db_key,
        "experiment": "within_subject_baseline",
        "num_kernels": num_kernels,
        "window_ms": window_ms,
        "overlap": overlap,
        "n_classes": n_classes,
        "n_subjects": len(all_subject_results),
        "n_seeds_per_subject": n_seeds,
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "mean": round(desc.get("mean", 0.0), 4),
            "std": round(desc.get("std", 0.0), 4),
            "median": round(desc.get("median", 0.0), 4),
            "ci95_lower": desc.get("ci95_lower", np.nan),
            "ci95_upper": desc.get("ci95_upper", np.nan),
            "n": len(accs),
        },
        "per_subject": all_subject_results,
    }

    json_path = os.path.join(output_dir, f"within_subject_{db_key}_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n{'='*70}", flush=True)
    print(f"  WITHIN-SUBJECT [{db_key}]: {desc.get('mean',0)*100:.2f}% ± {desc.get('std',0)*100:.2f}%", flush=True)
    print(f"  Saved: {json_path}", flush=True)
    print(f"{'='*70}\n", flush=True)
    return result


def parse_args():
    p = argparse.ArgumentParser(description="Within-subject baseline (same pipeline as LOSO)")
    p.add_argument("--db", type=str, default=None, choices=["db7", "db3", "db2"])
    p.add_argument("--all", action="store_true")
    p.add_argument("--num_kernels", type=int, default=5000)
    p.add_argument("--window_ms", type=int, default=config.DEFAULT_WINDOW_MS)
    p.add_argument("--overlap", type=float, default=config.DEFAULT_OVERLAP)
    p.add_argument("--per_subject_limit", type=int, default=5000)
    p.add_argument("--n_seeds", type=int, default=1,
                    help="عدد إعادات تقسيم 80/20 لكل شخص (لتقدير تباين التقسيم "
                         "فقط — هذا ليس بديلاً عن فحص حساسية بذرة نوى MiniROCKET، "
                         "انظر الملاحظة أسفل الملف)")
    p.add_argument("--output_dir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    databases = ["db7", "db3", "db2"] if args.all else [args.db or "db7"]
    for db_key in databases:
        try:
            run_within_subject_db(
                db_key=db_key, num_kernels=args.num_kernels,
                window_ms=args.window_ms, overlap=args.overlap,
                per_subject_limit=args.per_subject_limit,
                n_seeds=args.n_seeds, output_dir=args.output_dir,
            )
        except Exception as e:
            import traceback
            print(f"ERROR on {db_key}: {e}", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()

# ============================================================================
# ملاحظة مهمة حول --n_seeds:
# هذا الخيار يُنوّع فقط تقسيم 80/20 (أي نوافذ تذهب للتدريب/الاختبار)،
# وليس بذرة توليد نوى MiniROCKET نفسها (config.RANDOM_SEED ثابتة هنا عمداً
# لضمان مقارنة عادلة مع نتائج LOSO التي استخدمت نفس البذرة). لفحص حساسية
# بذرة النوى تحديداً، عدّل config.RANDOM_SEED في سطر الأوامر لهذا الملف أو
# لـ p01a_run_main_loso_benchmark.py وشغّل db7/raw مرتين إضافيتين ببذرتين مختلفتين
# (مثلاً 43 و44)، ثم قارن دقة LOSO الناتجة — هذا هو الفحص الفعلي للثبات
# تجاه اختيار النوى العشوائية.
# ============================================================================