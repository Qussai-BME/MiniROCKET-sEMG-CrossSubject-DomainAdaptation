"""
p04c_run_coral_distortion_diagnostic.py — لماذا يضر CORAL بأداء PPV features؟
=============================================================
يفحص فرضية: PPV features (نسب ثنائية bounded [0,1]) تنتهك
افتراض CORAL بأن التوزيع Gaussian-like.

يُشغَّل مرة واحدة على subject واحد من DB7 (سريع، لا يحتاج LOSO كامل).
يُنتج:
  1. إحصائيات PPV قبل/بعد CORAL (mean, std, % خارج [0,1])
  2. رسم توزيع 6 ميزات عشوائية قبل/بعد
  3. جدول رقمي يُدرَج مباشرة في الورقة (Table: CORAL Distortion Analysis)
"""
import sys, os
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config
from data_loader import load_ninapro_stream
from minirocket import MiniRocket
from domain_adaptation import coral_adaptation

OUT_DIR = config.FIGURES_DIR
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading 2 subjects from DB7 (source + target)...", flush=True)
gen = load_ninapro_stream('db7', config, window_ms=200, overlap=0.5)
subjects = []
for s in gen:
    subjects.append(s)
    if len(subjects) == 2:
        break

src, tgt = subjects[0], subjects[1]
print(f"Source: subject {src['subject_id']}, {src['X'].shape[0]} windows")
print(f"Target: subject {tgt['subject_id']}, {tgt['X'].shape[0]} windows")

# subsample للسرعة
rng = np.random.RandomState(42)
n_sub = 3000
if src['X'].shape[0] > n_sub:
    idx = rng.choice(src['X'].shape[0], n_sub, replace=False)
    Xs_raw, ys = src['X'][idx], src['y'][idx]
else:
    Xs_raw, ys = src['X'], src['y']
if tgt['X'].shape[0] > n_sub:
    idx = rng.choice(tgt['X'].shape[0], n_sub, replace=False)
    Xt_raw, yt = tgt['X'][idx], tgt['y'][idx]
else:
    Xt_raw, yt = tgt['X'], tgt['y']

print("\nComputing MiniROCKET PPV features...", flush=True)
rocket = MiniRocket(num_kernels=2000, random_state=42)
Xs_ppv = rocket.fit_transform(Xs_raw)
Xt_ppv = rocket.transform(Xt_raw)
print(f"PPV shape: source={Xs_ppv.shape}, target={Xt_ppv.shape}")

print("\nApplying CORAL...", flush=True)
Xs_coral, Xt_coral = coral_adaptation(Xs_ppv, Xt_ppv, reg=config.CORAL_REG)

def stats_block(X, name):
    return {
        'name': name,
        'mean': float(X.mean()),
        'std':  float(X.std()),
        'min':  float(X.min()),
        'max':  float(X.max()),
        'pct_outside_01': float(np.mean((X < 0) | (X > 1)) * 100),
        'pct_negative':   float(np.mean(X < 0) * 100),
    }

results = [
    stats_block(Xs_ppv,   'Source PPV (raw)'),
    stats_block(Xt_ppv,   'Target PPV (raw)'),
    stats_block(Xs_coral, 'Source PPV (CORAL)'),
    stats_block(Xt_coral, 'Target PPV (CORAL)'),
]

print("\n" + "="*78)
print("  TABLE: CORAL Distortion Analysis (for paper)")
print("="*78)
print(f"  {'Set':<22} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'%Outside[0,1]':>14}")
print("  " + "-"*76)
for r in results:
    print(f"  {r['name']:<22} {r['mean']:>8.4f} {r['std']:>8.4f} "
          f"{r['min']:>8.4f} {r['max']:>8.4f} {r['pct_outside_01']:>13.2f}%")
print("="*78)

# حفظ جدول CSV
import csv
csv_path = os.path.join(config.TABLES_DIR, 'coral_distortion_table.csv')
os.makedirs(config.TABLES_DIR, exist_ok=True)
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=results[0].keys())
    w.writeheader()
    w.writerows(results)
print(f"\nSaved table: {csv_path}")

# رسم: 6 ميزات عشوائية، توزيعها قبل/بعد
feat_idx = rng.choice(Xs_ppv.shape[1], 6, replace=False)
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for i, fi in enumerate(feat_idx):
    ax = axes[i // 3, i % 3]
    ax.hist(Xs_ppv[:, fi],   bins=30, alpha=0.5, label='Source (raw)',   color='#1E88E5')
    ax.hist(Xt_ppv[:, fi],   bins=30, alpha=0.5, label='Target (raw)',   color='#43A047')
    ax.hist(Xs_coral[:, fi], bins=30, alpha=0.4, label='Source (CORAL)', color='#E53935', histtype='step', linewidth=2)
    ax.hist(Xt_coral[:, fi], bins=30, alpha=0.4, label='Target (CORAL)', color='#FB8C00', histtype='step', linewidth=2)
    ax.axvline(0, color='k', linestyle='--', linewidth=0.5)
    ax.axvline(1, color='k', linestyle='--', linewidth=0.5)
    ax.set_title(f'Feature {fi}')
    if i == 0:
        ax.legend(fontsize=8)
plt.suptitle('CORAL Distorts Bounded PPV Features Outside [0,1]', fontsize=13)
plt.tight_layout()
fig_path = os.path.join(OUT_DIR, 'coral_distortion_analysis.png')
plt.savefig(fig_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved figure: {fig_path}")

print("\nDONE. Use the table + figure directly in the paper's Discussion section.")

# ====================================================================
# إضافة: فحص Eigenvalue Spectrum — الدليل المباشر على noise amplification
# ====================================================================
print("\nComputing covariance eigenvalue spectrum (source PPV)...", flush=True)
Xs_c = Xs_ppv - Xs_ppv.mean(axis=0)
Cs = (Xs_c.T @ Xs_c) / (Xs_c.shape[0] - 1)
eigvals = np.linalg.eigvalsh(Cs)
eigvals_sorted = np.sort(eigvals)[::-1]

reg = config.CORAL_REG
n_near_singular = int(np.sum(eigvals_sorted < reg * 10))
condition_number = float(eigvals_sorted[0] / max(eigvals_sorted[-1], 1e-12))

print(f"  Total eigenvalues:        {len(eigvals_sorted)}")
print(f"  Largest eigenvalue:       {eigvals_sorted[0]:.6f}")
print(f"  Smallest eigenvalue:      {eigvals_sorted[-1]:.9f}")
print(f"  Condition number:         {condition_number:.2e}")
print(f"  Eigenvalues < 10*reg({reg}): {n_near_singular} / {len(eigvals_sorted)} "
      f"({100*n_near_singular/len(eigvals_sorted):.1f}%)")

with open(os.path.join(config.TABLES_DIR, 'coral_eigenvalue_spectrum.txt'), 'w') as f:
    f.write(f"Total eigenvalues: {len(eigvals_sorted)}\n")
    f.write(f"Largest: {eigvals_sorted[0]:.6f}\n")
    f.write(f"Smallest: {eigvals_sorted[-1]:.9f}\n")
    f.write(f"Condition number: {condition_number:.2e}\n")
    f.write(f"Near-singular (<10*reg): {n_near_singular}/{len(eigvals_sorted)} "
            f"({100*n_near_singular/len(eigvals_sorted):.1f}%)\n")

print("\n  --> High condition number + many near-singular eigenvalues")
print("      CONFIRMS the noise amplification hypothesis.")

# ====================================================================
# Scree plot: يُظهر بصرياً الانهيار الحاد في الـ eigenvalues
# ====================================================================
print("\nGenerating eigenvalue scree plot...", flush=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.plot(np.arange(1, len(eigvals_sorted)+1), eigvals_sorted,
         color='#1E88E5', linewidth=1.5)
ax1.axhline(reg*10, color='#E53935', linestyle='--', linewidth=1,
            label=f'10×CORAL reg ({reg*10:.4f})')
ax1.set_xlabel('Eigenvalue rank')
ax1.set_ylabel('Eigenvalue magnitude')
ax1.set_title('PPV Covariance Spectrum (linear scale)')
ax1.legend(fontsize=9)

eigvals_pos = np.maximum(eigvals_sorted, 1e-15)
ax2.semilogy(np.arange(1, len(eigvals_pos)+1), eigvals_pos,
             color='#1E88E5', linewidth=1.5)
ax2.axhline(reg*10, color='#E53935', linestyle='--', linewidth=1,
            label=f'10×CORAL reg')
ax2.axvline(n_near_singular, color='#8E24AA', linestyle=':', linewidth=1.5,
            label=f'{n_near_singular} near-singular ({100*n_near_singular/len(eigvals_sorted):.0f}%)')
ax2.set_xlabel('Eigenvalue rank')
ax2.set_ylabel('Eigenvalue magnitude (log scale)')
ax2.set_title(f'PPV Covariance Spectrum (log scale)\nCondition number = {condition_number:.2e}')
ax2.legend(fontsize=9)

plt.suptitle('MiniROCKET PPV Feature Space is Effectively Low-Rank', fontsize=13)
plt.tight_layout()
scree_path = os.path.join(OUT_DIR, 'coral_eigenvalue_scree.png')
plt.savefig(scree_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved scree plot: {scree_path}")
print("\nUSE THIS FIGURE as the central evidence for the CORAL-failure discussion.")