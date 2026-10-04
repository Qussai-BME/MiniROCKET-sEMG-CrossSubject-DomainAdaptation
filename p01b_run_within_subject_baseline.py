"""
p01b_run_within_subject_baseline.py — Within-Subject Baseline
========================================================================================
(إعادة تصميم): النسخة V2 كانت تُقسّم *نوافذ* كل شخص 80/20
مباشرة (stratified على مستوى النافذة). بما أن النوافذ تتداخل 50%، نافذتان
متجاورتان تتشاركان نصف عيناتهما الخام — فتقسيم على مستوى النافذة يُسرّب
عينات خام من train إلى test (والعكس). **رقم 4-6× within/LOSO في المخطوطة
الحالية مبني على هذا التقسيم المُسرِّب ويُعتبر من الآن "قديم/legacy" حتى
تُعاد التجربة.**

الإصلاح: التقسيم الآن على مستوى *التكرار الكامل* (repetition)، عبر
subject_split.split_by_repetition — تماماً كما في تقسيم
calibration/final-test لـ LOSO ()، بحيث لا تكرار واحد يظهر
جزئياً في train وجزئياً في test.

تحسينات إضافية لتتوافق مع بقية التدقيق:
  - يستخدم نفس feature_extractors.get_feature_extractor
    (افتراضياً "canonical")، حتى تبقى المقارنة مع Table 2 الجديدة عادلة.
  - balanced_accuracy وmacro_f1 تُحسبان أيضاً.

الاستخدام:
  python p01b_run_within_subject_baseline.py --db db7
  python p01b_run_within_subject_baseline.py --all
"""
import sys, os, json, time, argparse, warnings
from datetime import datetime
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import config
from data_loader import load_ninapro_stream, validate_data_path, get_n_classes
from feature_extractors import get_feature_extractor
from subject_split import split_by_repetition
from statistical_utils import descriptive_stats, random_baseline_from_labels, ratio_vs_random

warnings.filterwarnings('ignore')


def run_within_subject_fold(X, y, rep_id, num_kernels, feature_extractor,
                            test_fraction=0.2, seed=42):
    """
    التقسيم يتم على مستوى التكرار (rep_id) عبر
    split_by_repetition، وليس على مستوى النافذة. mask_kept = train،
    mask_held_out = test (held_out_fraction=test_fraction).
    """
    X = X.astype(np.float32)

    mask_train, mask_test, split_warnings = split_by_repetition(
        y, rep_id, held_out_fraction=test_fraction, seed=seed,
        min_reps_per_class_each_side=1,
    )
    X_train, y_train = X[mask_train], y[mask_train]
    X_test,  y_test  = X[mask_test],  y[mask_test]

    if len(y_test) == 0 or len(y_train) == 0:
        return {"error": "empty train or test partition after "
                          "repetition-level split", "accuracy": 0.0}

    rocket = get_feature_extractor(feature_extractor, num_kernels=num_kernels,
                                   random_state=seed)
    X_train_ppv = rocket.fit_transform(X_train)
    X_test_ppv  = rocket.transform(X_test)

    from domain_adaptation import BatchedStandardScaler
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.metrics import balanced_accuracy_score, f1_score
    scaler = BatchedStandardScaler(batch_size=5000)
    X_train_s = scaler.fit_transform(X_train_ppv)
    X_test_s  = scaler.transform(X_test_ppv)

    clf = None
    for kwargs_try in (dict(alphas=np.logspace(-3, 6, 25), scoring='accuracy'),
                       dict(alphas=np.logspace(-3, 6, 25))):
        try:
            clf = RidgeClassifierCV(**kwargs_try)
            clf.fit(X_train_s, y_train)
            break
        except TypeError:
            clf = None
    if clf is None:
        raise RuntimeError("RidgeClassifierCV construction failed.")

    y_pred = clf.predict(X_test_s)
    accuracy = float(np.mean(y_pred == y_test))
    bal_acc  = float(balanced_accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average='macro', zero_division=0))

    n_classes_present = len(np.unique(y))
    random_baseline_pct = random_baseline_from_labels(y_test)
    ratio = ratio_vs_random(accuracy, random_baseline_pct)

    return {
        "accuracy": accuracy,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "n_train_samples": int(len(y_train)),
        "n_test_samples": int(len(y_test)),
        "n_train_repetitions": int(len(np.unique(rep_id[mask_train]))),
        "n_test_repetitions": int(len(np.unique(rep_id[mask_test]))),
        "n_classes": int(n_classes_present),
        "random_baseline_pct": float(random_baseline_pct),
        "ratio_vs_random": float(ratio),
        "split_warnings": split_warnings,
        "split_unit": "repetition",       # marker (was "window")
    }


def run_within_subject_db(db_key, num_kernels=5000, window_ms=None, overlap=None,
                            per_subject_limit=5000, n_seeds=1, output_dir=None,
                            feature_extractor=None, test_fraction=None):
    window_ms = window_ms or config.DEFAULT_WINDOW_MS
    overlap = overlap or config.DEFAULT_OVERLAP
    output_dir = output_dir or config.RESULTS_DIR
    feature_extractor = feature_extractor or config.FEATURE_EXTRACTOR
    test_fraction = (config.WITHIN_SUBJECT_TEST_FRACTION
                     if test_fraction is None else test_fraction)
    os.makedirs(output_dir, exist_ok=True)

    validate_data_path(db_key, config)
    n_classes = get_n_classes(db_key, config)

    all_subject_results = []
    stream = load_ninapro_stream(db_key, config, window_ms, overlap)

    for subj in stream:
        sid = subj["subject_id"]
        X, y, rep_id = subj["X"], subj["y"], subj["rep_id"]
        # أي تصغير لحجم العينة يجب أن يبقى على مستوى
        # التكرار الكامل أيضاً (لا نختار نوافذ فردية عشوائياً، فقد يُبقي
        # ذلك جزءاً من تكرار ويحذف باقيه، فيُفسد لاحقاً حساب rep_id
        # الصحيح لأي تكرار متبقٍّ جزئياً -- لذا هنا نُبقي على كل النوافذ
        # إن كانت أقل من الحد، وإلا نُسقط تكرارات كاملة عشوائياً حتى
        # نصل تحت الحد).
        if len(y) > per_subject_limit:
            rng = np.random.RandomState(42)
            reps_avail = np.unique(rep_id)
            rng.shuffle(reps_avail)
            keep_mask = np.zeros(len(y), dtype=bool)
            running_total = 0
            for r in reps_avail:
                r_mask = (rep_id == r)
                if running_total + r_mask.sum() > per_subject_limit and running_total > 0:
                    continue
                keep_mask |= r_mask
                running_total += r_mask.sum()
            X, y, rep_id = X[keep_mask], y[keep_mask], rep_id[keep_mask]

        seed_accs = []
        fold_details = []
        for seed_offset in range(n_seeds):
            t0 = time.time()
            fold = run_within_subject_fold(
                X, y, rep_id, num_kernels, feature_extractor,
                test_fraction=test_fraction, seed=42 + seed_offset,
            )
            fold["time_sec"] = round(time.time() - t0, 2)
            fold["split_seed"] = 42 + seed_offset
            fold_details.append(fold)
            if "error" not in fold:
                seed_accs.append(fold["accuracy"])
            print(f"  [{db_key}] Subject {sid}, split_seed={42+seed_offset}: "
                  f"acc={fold.get('accuracy',0)*100:.2f}% "
                  f"({fold.get('time_sec',0):.1f}s)", flush=True)

        all_subject_results.append({
            "subject_id": sid,
            "accuracy_mean": float(np.mean(seed_accs)) if seed_accs else 0.0,
            "accuracy_std": float(np.std(seed_accs)) if len(seed_accs) > 1 else 0.0,
            "n_splits": n_seeds,
            "per_split_accuracy": seed_accs,
            "fold_details": fold_details,
        })

    accs = [r["accuracy_mean"] for r in all_subject_results]
    desc = descriptive_stats(accs) if accs else {}

    result = {
        "database": db_key,
        "experiment": "within_subject_baseline_v3_repetition_split",
        "split_unit": "repetition",                # was "window" in V2 — LEGACY
        "test_fraction": test_fraction,
        "feature_extractor": feature_extractor,
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

    json_path = os.path.join(output_dir, f"within_subject_v3_{db_key}_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n{'='*70}", flush=True)
    print(f"  WITHIN-SUBJECT (repetition-split) [{db_key}]: "
          f"{desc.get('mean',0)*100:.2f}% ± {desc.get('std',0)*100:.2f}%", flush=True)
    print(f"  Saved: {json_path}", flush=True)
    print(f"{'='*70}\n", flush=True)
    return result


def parse_args():
    p = argparse.ArgumentParser(
        description="Within-subject baseline (repetition-level split)")
    p.add_argument("--db", type=str, default=None, choices=["db7", "db3", "db2"])
    p.add_argument("--all", action="store_true")
    p.add_argument("--extractor", choices=config.FEATURE_EXTRACTOR_OPTIONS,
                   default=config.FEATURE_EXTRACTOR)
    p.add_argument("--test_fraction", type=float,
                   default=config.WITHIN_SUBJECT_TEST_FRACTION)
    p.add_argument("--num_kernels", type=int, default=5000)
    p.add_argument("--window_ms", type=int, default=config.DEFAULT_WINDOW_MS)
    p.add_argument("--overlap", type=float, default=config.DEFAULT_OVERLAP)
    p.add_argument("--per_subject_limit", type=int, default=5000)
    p.add_argument("--n_seeds", type=int, default=1,
                    help="عدد إعادات تقسيم التكرارات لكل شخص (لتقدير تباين "
                         "التقسيم فقط)")
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
                feature_extractor=args.extractor,
                test_fraction=args.test_fraction,
            )
        except Exception as e:
            import traceback
            print(f"ERROR on {db_key}: {e}", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()

# ============================================================================
# — ملاحظة توثيقية:
# نتيجة V2 القديمة (تقسيم على مستوى النافذة، أرشيف within_subject_{db}_results.json
# إن وُجد) يجب اعتبارها "legacy" في أي جدول أو نص بالمخطوطة حتى يُعاد
# تشغيل هذا الملف V3 وتحل نتائجه الجديدة (within_subject_v3_{db}_results.json)
# محلها. نسبة 4-6× within/LOSO المذكورة حالياً في القسم 3.x من المخطوطة
# مبنية على النتيجة القديمة المُسرِّبة ولا يجوز نقلها دون إعادة التشغيل.
# ============================================================================
