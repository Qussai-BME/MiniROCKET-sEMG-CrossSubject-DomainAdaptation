"""
p08_run_multiseed_sweep.py — 
========================================================================
يُكرّر تجربة LOSO الرئيسية (p01a) عبر config.SEED_LIST بدل الاعتماد على
seed=42 وحده، ثم يُجمّع تباين الـ seed نفسه (بجانب تباين الأشخاص الذي
تقيسه إحصاءات p06 أصلاً).

كل seed يُنتج ملف results منفصل تماماً (الوسم يتضمن seedN، انظر
p01a)، فلا خطر كتابة فوق أي نتيجة.

الاستخدام:
    python p08_run_multiseed_sweep.py --db db7 --method coral
    python p08_run_multiseed_sweep.py --all
"""
import sys, os, json, argparse
from datetime import datetime
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import config
from optimized_pipeline import run_loso_shared
from statistical_utils import descriptive_stats


def parse_args():
    p = argparse.ArgumentParser(
        description="multi-seed sweep of the main LOSO experiment")
    p.add_argument('--db', choices=['db7', 'db3', 'db2'], default=None)
    p.add_argument('--method', choices=config.METHODS, default=None)
    p.add_argument('--all', action='store_true',
                   help='كل الـ DBs والطرق × كل seeds في config.SEED_LIST')
    p.add_argument('--extractor', choices=config.FEATURE_EXTRACTOR_OPTIONS,
                   default=config.FEATURE_EXTRACTOR)
    p.add_argument('--seeds', type=str, default=None,
                   help="قائمة seeds مفصولة بفواصل، افتراضياً "
                        "config.SEED_LIST كاملة")
    p.add_argument('--num_kernels', type=int, default=10_000)
    p.add_argument('--max_train_samples', type=int, default=50_000)
    p.add_argument('--per_subject_limit', type=int, default=5_000)
    p.add_argument('--max_subjects', type=int, default=None,
                   help='لتسريع اختبار سريع قبل التشغيل الكامل')
    p.add_argument('--output_dir', type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    seeds = ([int(s) for s in args.seeds.split(',')]
            if args.seeds else list(config.SEED_LIST))

    if args.all:
        databases = ['db7', 'db3', 'db2']
        methods   = config.METHODS
    else:
        databases = [args.db]     if args.db     else ['db7']
        methods   = [args.method] if args.method else config.METHODS

    output_dir = args.output_dir or config.RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    print('=' * 72)
    print(f'  MULTI-SEED SWEEP')
    print(f'  DBs={databases}  Methods={methods}  Seeds={seeds}  '
          f'Extractor={args.extractor}')
    print('=' * 72)

    aggregate = {}   # key = f"{db}_{method}" -> list of per-seed means

    for db_key in databases:
        for method in methods:
            key = f"{db_key}_{method}"
            aggregate[key] = {'accuracy': [], 'balanced_accuracy': [],
                              'macro_f1': [], 'seeds': []}
        for seed in seeds:
            print(f"\n>>> {db_key}  seed={seed}  methods={methods}", flush=True)
            # Use the audited shared-feature pipeline once per db/seed; all
            # five methods reuse the same fitted feature representation.
            os.environ.setdefault('ALLOW_CACHE_ONLY', '1')
            all_res = run_loso_shared(
                db_key=db_key, methods=methods,
                num_kernels=args.num_kernels, output_dir=output_dir,
                seed=seed, feature_extractor=args.extractor,
                max_train_samples=args.max_train_samples,
                per_subject_limit=args.per_subject_limit,
                max_subjects=args.max_subjects, resume=True,
                fit_mode='full_train', rocket_n_jobs=1,
            )
            for method in methods:
                s = all_res[method]['summary']
                key = f"{db_key}_{method}"
                aggregate[key]['accuracy'].append(s['mean'])
                aggregate[key]['balanced_accuracy'].append(s['balanced_accuracy_mean'])
                aggregate[key]['macro_f1'].append(s['macro_f1_mean'])
                aggregate[key]['seeds'].append(seed)

    # ── ملخص تباين الـ seed ─────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  SEED-LEVEL VARIANCE SUMMARY (across config.SEED_LIST)")
    print(f"  {'experiment':<28} {'acc_mean':>9} {'acc_std(seed)':>14} "
          f"{'bal_acc_mean':>13} {'macro_f1_mean':>14}")
    summary_out = {}
    for key, vals in aggregate.items():
        acc_desc = descriptive_stats(vals['accuracy']) if len(vals['accuracy']) > 1 else {}
        acc_mean = float(np.mean(vals['accuracy'])) if vals['accuracy'] else 0.0
        acc_std_seed = float(np.std(vals['accuracy'])) if len(vals['accuracy']) > 1 else 0.0
        bal_mean = float(np.mean(vals['balanced_accuracy'])) if vals['balanced_accuracy'] else 0.0
        f1_mean  = float(np.mean(vals['macro_f1'])) if vals['macro_f1'] else 0.0
        summary_out[key] = {
            'seeds': vals['seeds'],
            'accuracy_per_seed': vals['accuracy'],
            'accuracy_mean_across_seeds': round(acc_mean, 4),
            'accuracy_std_across_seeds': round(acc_std_seed, 4),
            'balanced_accuracy_mean_across_seeds': round(bal_mean, 4),
            'macro_f1_mean_across_seeds': round(f1_mean, 4),
        }
        print(f"  {key:<28} {acc_mean:>9.4f} {acc_std_seed:>14.4f} "
              f"{bal_mean:>13.4f} {f1_mean:>14.4f}")
    print('=' * 72)

    out_path = os.path.join(
        output_dir,
        f"MULTISEED_SUMMARY_{'-'.join(databases)}_"
        f"{'_'.join(map(str, seeds))}.json")
    with open(out_path, 'w') as f:
        json.dump({
            'databases': databases, 'seeds': seeds, 'extractor': args.extractor,
            'timestamp': datetime.now().isoformat(),
            'results': summary_out,
        }, f, indent=2)
    print(f"\n  Multi-seed summary -> {out_path}")


if __name__ == '__main__':
    main()