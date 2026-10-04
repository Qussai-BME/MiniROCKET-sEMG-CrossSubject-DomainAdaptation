"""
make_figure5_and_table_s13.py -- Figure 5 and Supplementary Table S13 of the article
====================================================================================
Runs the five synthetic Gaussian problems defined in tools/audit_coral_synthetic_figure5.py (unchanged)
and writes
    outputs/tables/TableS13_synthetic_coral.csv        (dimension, sample sizes, offsets, scales, seed, regularization,
                                                        covariance distance before/after, reduction)
    outputs/figures/figure5_synthetic_coral.{png,pdf}

Usage (from any directory)
    python paper_reproduction/make_figure5_and_table_s13.py
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location(
    "audit_coral_synthetic_figure5", os.path.join(ROOT, "tools", "audit_coral_synthetic_figure5.py"))
syn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(syn)


def main():
    rows = [syn.run_case(c) for c in syn.CASES]
    df = pd.DataFrame(rows).rename(columns={
        "d": "dimension", "ns": "n_source", "nt": "n_target", "shift": "mean_offset",
        "ss": "scale_source", "st": "scale_target", "before": "cov_distance_before",
        "after": "cov_distance_after", "red": "reduction"})
    df["reduction_pct"] = 100 * df["reduction"]
    df = df.drop(columns="reduction")
    for _, r in df.iterrows():
        print(f"d={int(r.dimension):3d}  before {r.cov_distance_before:10.4f} -> after {r.cov_distance_after:10.4f}"
              f"  reduction {r.reduction_pct:5.2f} %")

    tdir = os.path.join(ROOT, "outputs", "tables")
    fdir = os.path.join(ROOT, "outputs", "figures")
    os.makedirs(tdir, exist_ok=True)
    os.makedirs(fdir, exist_ok=True)
    df.to_csv(os.path.join(tdir, "TableS13_synthetic_coral.csv"), index=False, float_format="%.6g")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df.cov_distance_before, 0.4, color="#9aa5b1", label="Before")
    ax.bar(x + 0.2, df.cov_distance_after, 0.4, color="#d95f02", label="After CORAL")
    for i, r in df.iterrows():
        ax.text(i + 0.2, r.cov_distance_after * 1.15, f"\u2212{r.reduction_pct:.0f}%", ha="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"d={int(d)}" for d in df.dimension])
    ax.set_ylabel("Frobenius distance (log)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fdir, f"figure5_synthetic_coral.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote TableS13_synthetic_coral.csv and figure5_synthetic_coral.{png,pdf}")
    if not (df.reduction_pct > 0).all():
        sys.exit(1)


if __name__ == "__main__":
    main()
