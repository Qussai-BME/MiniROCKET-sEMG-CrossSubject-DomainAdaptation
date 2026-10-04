"""
audit_coral_synthetic_figure5.py -- synthetic Gaussian verification of CORAL (Figure 5 / Table S13)
===========================================================================================
Reproduces EXACTLY the five synthetic problems reported in Section 3.3, Figure 5 and
Supplementary Table S13 of the manuscript.

Recipe (fully specified so that independent re-implementations give the same numbers)
  * For each case: random covariances  C = A A^T + 0.01 I,  A_ij ~ N(0, scale^2)
      Cs from scale_s, Ct from scale_t (drawn in this order from RandomState(seed)).
  * Source  ~ N(0,        Cs), n_source samples.
    Target calibration ~ N(shift*1, Ct), n_target samples        (used to fit CORAL)
    Held-out target    ~ N(shift*1, Ct), n_target INDEPENDENT samples (used only to measure distance)
  * CORAL = repository function domain_adaptation.coral_adaptation(Xs, X_cal, X_heldout, reg=1e-3):
        only the source is whitened/re-coloured; the held-out target is returned unchanged.
  * Distance = Frobenius norm of the difference of the sample covariance matrices
        (np.cov, n-1 denominator) between the (adapted) source and the held-out target.
  * Reduction = 1 - distance_after / distance_before.

Usage (from the project root):
    python tools/audit_coral_synthetic_figure5.py
    python tools/audit_coral_synthetic_figure5.py --out_csv outputs/tables/S13_synthetic_coral.csv

Expected output (numpy RandomState streams are version-stable; BLAS may change the 5th digit):
    d=  8  before   96.4995 -> after    1.2784  reduction 98.68 %
    d= 20  before  261.9336 -> after   20.0740  reduction 92.34 %
    d=100  before 8536.7458 -> after 1260.7491  reduction 85.23 %
    d=200  before 5342.4253 -> after 1148.9452  reduction 78.49 %
    d=100  before 12599.0697 -> after 2392.3270  reduction 81.01 %
The script exits with status 1 if any case does not show a reduction.
"""
import os, sys, json, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import domain_adaptation as da   # noqa: E402

CASES = [  # d, n_source, n_target, mean offset, scale_s, scale_t, seed
    dict(d=8,   ns=20000, nt=20000, shift=1.0, ss=1.0, st=1.6, seed=0),
    dict(d=20,  ns=20000, nt=5000,  shift=2.0, ss=1.0, st=1.5, seed=1),
    dict(d=100, ns=20000, nt=5000,  shift=1.0, ss=1.0, st=2.5, seed=2),
    dict(d=200, ns=20000, nt=5000,  shift=0.5, ss=0.5, st=1.2, seed=3),
    dict(d=100, ns=20000, nt=3000,  shift=3.0, ss=0.3, st=3.0, seed=4),
]
REG = 1e-3


def spd(d, scale, rng):
    A = rng.standard_normal((d, d)) * scale
    return A @ A.T + np.eye(d) * 1e-2


def cov_distance(a, b):
    return float(np.linalg.norm(np.cov(a, rowvar=False) - np.cov(b, rowvar=False), 'fro'))


def run_case(c):
    rng = np.random.RandomState(c['seed']); d = c['d']
    Cs = spd(d, c['ss'], rng); Ct = spd(d, c['st'], rng)
    Xs = rng.multivariate_normal(np.zeros(d), Cs, c['ns'])
    Xcal = rng.multivariate_normal(np.ones(d) * c['shift'], Ct, c['nt'])
    Xho = rng.multivariate_normal(np.ones(d) * c['shift'], Ct, c['nt'])   # held-out target sample
    before = cov_distance(Xs, Xho)
    Xs_a, _, Xho_a = da.coral_adaptation(Xs, Xcal, Xho, reg=REG)
    assert np.allclose(Xho_a, Xho.astype(np.float32)), 'target must be returned unchanged'
    after = cov_distance(Xs_a, Xho_a)
    return dict(c, reg=REG, before=before, after=after, red=1 - after / before)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_csv', default=None)
    ap.add_argument('--out_json', default=None)
    a = ap.parse_args()
    rows = []
    for c in CASES:
        r = run_case(c); rows.append(r)
        print(f"d={r['d']:3d}  before {r['before']:10.4f} -> after {r['after']:10.4f}  reduction {100*r['red']:5.2f} %")
    ok = all(r['red'] > 0 for r in rows)
    print('\nreduction range: %.1f %% - %.1f %%   ->  %s' % (100*min(r['red'] for r in rows),
          100*max(r['red'] for r in rows), 'PASS' if ok else 'FAIL'))
    if a.out_csv:
        import csv
        os.makedirs(os.path.dirname(os.path.abspath(a.out_csv)), exist_ok=True)
        with open(a.out_csv, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    if a.out_json:
        json.dump(rows, open(a.out_json, 'w'), indent=1)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
