"""
p13_article_tables_and_figures.py
=================================
Builds every table and data-derived figure of the article from the per-subject,
per-seed result files in ``outputs/results/`` (``results_<db>_<method>__canonical__seed<S>__shared.json``).

Statistical design (article Section 2.7)
  * unit of analysis = held-out subject; every metric is first averaged over the five seeds;
  * conditions compared: Raw, CORAL, TCA, SA (Centering is identical to Raw and is excluded);
  * Friedman omnibus test with Iman-Davenport F correction;
  * two-sided paired Wilcoxon signed-rank tests for the six pairwise contrasts, Holm-Sidak adjusted
    within each database and metric;
  * effect sizes: mean paired difference (pp) with a 95 % percentile bootstrap interval
    (10,000 resamples of subjects), paired d_z, and the number of subjects that improve / worsen.

Usage (from any directory)
    python paper_reproduction/p13_article_tables_and_figures.py                # tables + figures
    python paper_reproduction/p13_article_tables_and_figures.py --no_figures   # tables only
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SEEDS = [42, 123, 2024, 7, 999]
DBS = ["db7", "db3", "db2"]
DB_LABEL = {"db7": "NinaPro DB7", "db3": "NinaPro DB3", "db2": "NinaPro DB2"}
METHODS = ["raw", "coral", "tca", "sa"]          # inferential conditions
ALL_METHODS = METHODS + ["centering"]
M_LABEL = {"raw": "Raw", "coral": "CORAL", "tca": "TCA", "sa": "SA", "centering": "Centering"}
METRICS = [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced acc."), ("macro_f1", "Macro-F1")]
BOOT_N = 10000
BOOT_SEED = 20260101
DB7_AMPUTEES = (21, 22)


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------
def load_all(results_dir):
    """Return {(db, method): {seed: {subject_id: record}}}."""
    data = {}
    for db in DBS:
        for m in ALL_METHODS:
            for s in SEEDS:
                path = os.path.join(results_dir, f"results_{db}_{m}__canonical__seed{s}__shared.json")
                if not os.path.exists(path):
                    raise FileNotFoundError(path)
                with open(path, encoding="utf8") as fh:
                    j = json.load(fh)
                data.setdefault((db, m), {})[s] = {r["subject_id"]: r for r in j["per_subject"]}
    return data


def subject_ids(data, db):
    ids = sorted(data[(db, "raw")][SEEDS[0]].keys())
    for m in ALL_METHODS:
        for s in SEEDS:
            assert sorted(data[(db, m)][s].keys()) == ids, f"subject mismatch {db} {m} {s}"
    return ids


def seed_avg(data, db, m, key):
    """Seed-averaged scalar metric per subject -> array (n_subjects,) in the order of subject_ids."""
    ids = subject_ids(data, db)
    return np.array([np.mean([data[(db, m)][s][i][key] for s in SEEDS]) for i in ids])


def n_classes(data, db):
    return len(data[(db, "raw")][SEEDS[0]][subject_ids(data, db)[0]]["f1_per_class"])


# --------------------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------------------
def boot_ci(x, rng):
    idx = rng.randint(0, len(x), size=(BOOT_N, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm_sidak(p):
    """Holm-Sidak step-down adjusted p-values."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        a = 1.0 - (1.0 - p[idx]) ** (m - rank)
        running = max(running, a)
        adj[idx] = min(1.0, running)
    return adj


def friedman_iman_davenport(mat):
    """mat: (N subjects, k conditions), higher = better. Returns chi2_F, p_ID, mean ranks (1 = best)."""
    n, k = mat.shape
    ranks = np.vstack([stats.rankdata(-row) for row in mat])
    mr = ranks.mean(axis=0)
    chi2 = 12.0 * n / (k * (k + 1)) * np.sum((mr - (k + 1) / 2.0) ** 2)
    denom = n * (k - 1) - chi2
    if denom <= 0:
        return chi2, 0.0, mr
    f = (n - 1) * chi2 / denom
    p = float(stats.f.sf(f, k - 1, (k - 1) * (n - 1)))
    return float(chi2), p, mr


def wilcoxon_p(a, b):
    d = a - b
    if np.allclose(d, 0):
        return 1.0
    return float(stats.wilcoxon(a, b, alternative="two-sided").pvalue)


def fmt_p(p):
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


# --------------------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------------------
def table1_volumes(data):
    rows = []
    for db in DBS:
        recs = [data[(db, "raw")][s][i] for s in SEEDS for i in subject_ids(data, db)]
        recs_c = [data[(db, "coral")][s][i] for s in SEEDS for i in subject_ids(data, db)]
        K = n_classes(data, db)
        rows.append({
            "database": DB_LABEL[db], "held_out_subjects": len(subject_ids(data, db)), "classes": K,
            "chance_pct": round(100.0 / K, 1),
            "source_subjects_per_fold_min": int(min(r["n_train_subjects"] for r in recs)),
            "source_subjects_per_fold_max": int(max(r["n_train_subjects"] for r in recs)),
            "source_windows_per_fold_mean": round(np.mean([r["n_train_samples"] for r in recs])),
            "calibration_windows_mean": round(np.mean([r["n_calibration_samples_used"] for r in recs_c])),
            "calibration_windows_min": int(min(r["n_calibration_samples_used"] for r in recs_c)),
            "calibration_windows_max": int(max(r["n_calibration_samples_used"] for r in recs_c)),
            "final_test_windows_mean": round(np.mean([r["n_test_samples"] for r in recs])),
        })
    return pd.DataFrame(rows)


def table2_main(data, rng):
    rows = []
    for db in DBS:
        K = n_classes(data, db)
        chance = 1.0 / K
        for m in METHODS:
            acc = seed_avg(data, db, m, "accuracy")
            ba = seed_avg(data, db, m, "balanced_accuracy")
            f1 = seed_avg(data, db, m, "macro_f1")
            lo, hi = boot_ci(acc * 100, rng)
            rows.append({
                "database": DB_LABEL[db], "method": M_LABEL[m], "n_subjects": len(acc), "classes": K,
                "chance_pct": round(100 * chance, 2),
                "accuracy_mean": acc.mean() * 100, "accuracy_sd": acc.std(ddof=1) * 100,
                "accuracy_ci95_low": lo, "accuracy_ci95_high": hi,
                "balanced_acc_mean": ba.mean() * 100, "balanced_acc_sd": ba.std(ddof=1) * 100,
                "macro_f1_mean": f1.mean() * 100, "macro_f1_sd": f1.std(ddof=1) * 100,
                "x_chance": acc.mean() / chance, "subjects_above_chance": int((acc > chance).sum()),
            })
    return pd.DataFrame(rows)


def stats_tables(data, rng):
    """Friedman (Table 3), full pairwise (S7-S9 equivalents) and effects vs Raw (Table 4)."""
    fried, pair, eff = [], [], []
    for db in DBS:
        for key, lab in METRICS:
            mat = np.column_stack([seed_avg(data, db, m, key) for m in METHODS])
            chi2, p_id, mr = friedman_iman_davenport(mat)
            fried.append({"database": DB_LABEL[db], "metric": lab, "N": mat.shape[0], "chi2_F": chi2,
                          "iman_davenport_p": p_id, **{f"rank_{m}": r for m, r in zip(METHODS, mr)}})
            # six pairwise contrasts
            pairs = [(a, b) for i, a in enumerate(METHODS) for b in METHODS[i + 1:]]
            raw_p = np.array([wilcoxon_p(mat[:, METHODS.index(a)], mat[:, METHODS.index(b)]) for a, b in pairs])
            adj = holm_sidak(raw_p)
            for (a, b), p, pa in zip(pairs, raw_p, adj):
                d = (mat[:, METHODS.index(a)] - mat[:, METHODS.index(b)]) * 100
                lo, hi = boot_ci(d, rng)
                pair.append({"database": DB_LABEL[db], "metric": lab, "contrast": f"{M_LABEL[a]} vs {M_LABEL[b]}",
                             "mean_diff_pp": d.mean(), "ci95_low": lo, "ci95_high": hi,
                             "p_raw": p, "p_holm_sidak": pa, "d_z": d.mean() / d.std(ddof=1),
                             "n_improve": int((d > 0).sum()), "n_worsen": int((d < 0).sum())})
            # restricted family: three contrasts against Raw only (supplement)
            p3 = np.array([wilcoxon_p(mat[:, METHODS.index(m)], mat[:, 0]) for m in METHODS[1:]])
            a3 = holm_sidak(p3)
            for m, p, pa in zip(METHODS[1:], p3, a3):
                eff.append({"database": DB_LABEL[db], "metric": lab, "method": M_LABEL[m],
                            "p_holm_sidak_3contrasts": pa})
    fried = pd.DataFrame(fried)
    pair = pd.DataFrame(pair)
    restricted = pd.DataFrame(eff)
    # Table 4 = contrasts "X vs Raw" -> sign so that positive favours the UDA method
    rows = []
    for db in DBS:
        for m in METHODS[1:]:
            row = {"database": DB_LABEL[db], "method": M_LABEL[m]}
            for key, lab in METRICS:
                sel = pair[(pair.database == DB_LABEL[db]) & (pair.metric == lab) &
                           (pair.contrast == f"Raw vs {M_LABEL[m]}")].iloc[0]
                sign = -1.0                                   # contrast is Raw - method
                tag = {"Accuracy": "acc", "Balanced acc.": "ba", "Macro-F1": "f1"}[lab]
                row[f"delta_{tag}_pp"] = sign * sel.mean_diff_pp
                if tag == "acc":
                    row["delta_acc_ci95_low"] = sign * sel.ci95_high
                    row["delta_acc_ci95_high"] = sign * sel.ci95_low
                    row["d_z"] = sign * sel.d_z
                    row["subjects_improve"] = int(sel.n_worsen)
                    row["subjects_worsen"] = int(sel.n_improve)
                row[f"p_adj_{tag}"] = sel.p_holm_sidak
            rows.append(row)
    return fried, pair, pd.DataFrame(rows), restricted


def per_subject_table(data):
    rows = []
    for db in DBS:
        ids = subject_ids(data, db)
        cols = {}
        for m in METHODS:
            for key, lab in METRICS:
                cols[(m, key)] = seed_avg(data, db, m, key) * 100
        for n, sid in enumerate(ids):
            row = {"database": DB_LABEL[db], "subject_id": sid}
            for m in METHODS:
                for key, tag in (("accuracy", "acc"), ("balanced_accuracy", "ba"), ("macro_f1", "f1")):
                    row[f"{m}_{tag}"] = cols[(m, key)][n]
            rows.append(row)
    return pd.DataFrame(rows)


def coral_shift_table(data):
    rows, per_eval = [], []
    for db in DBS:
        ids = subject_ids(data, db)
        red_eval, cb, ca, mb, ma = [], [], [], [], []
        for i in ids:
            vals = [data[(db, "coral")][s][i]["domain_shift"] for s in SEEDS]
            cb.append(np.mean([v["cov_before"] for v in vals]))
            ca.append(np.mean([v["cov_after"] for v in vals]))
            mb.append(np.mean([v["mmd_before"] for v in vals]))
            ma.append(np.mean([v["mmd_after"] for v in vals]))
            for v in vals:
                red_eval.append(100 * (1 - v["cov_after"] / v["cov_before"]))
        cb, ca = np.array(cb), np.array(ca)
        subj_red = 100 * (1 - ca / cb)
        rows.append({"database": DB_LABEL[db], "fold_seed_evaluations": len(red_eval),
                     "cov_dist_before": cb.mean(), "cov_dist_after": ca.mean(),
                     "mean_reduction_pct": subj_red.mean(), "median_reduction_pct": float(np.median(subj_red)),
                     "min_reduction_pct_over_evaluations": min(red_eval),
                     "evaluations_with_reduction": int(sum(r > 0 for r in red_eval)),
                     "mmd_before": float(np.mean(mb)), "mmd_after": float(np.mean(ma))})
    return pd.DataFrame(rows)


def subgroup_table(data):
    rows = []
    ids7 = subject_ids(data, "db7")
    for name, db, sel in [("DB7 intact (IDs 1-20)", "db7", [i for i in ids7 if i not in DB7_AMPUTEES]),
                          ("DB7 amputee, ID 21", "db7", [21]), ("DB7 amputee, ID 22", "db7", [22]),
                          ("DB3 amputees (all 11)", "db3", subject_ids(data, "db3"))]:
        ids = subject_ids(data, db)
        mask = np.array([i in sel for i in ids])
        row = {"group": name, "n": int(mask.sum()), "chance_pct": round(100.0 / n_classes(data, db), 1)}
        for m in METHODS:
            row[M_LABEL[m]] = (seed_avg(data, db, m, "accuracy")[mask].mean()) * 100
        rows.append(row)
    return pd.DataFrame(rows)


def prediction_balance(data):
    rows = []
    for db in DBS:
        ids = subject_ids(data, db)
        K = n_classes(data, db)
        for m in METHODS:
            top, never, ent = [], [], []
            for s in SEEDS:
                for i in ids:
                    cm = np.asarray(data[(db, m)][s][i]["confusion_matrix"], dtype=float)
                    pred = cm.sum(axis=0)
                    tot = pred.sum()
                    if tot == 0:
                        continue
                    q = pred / tot
                    top.append(q.max() * 100)
                    never.append(100.0 * (pred == 0).sum() / K)
                    nz = q[q > 0]
                    ent.append(float(-(nz * np.log(nz)).sum() / np.log(K)))
            rows.append({"database": DB_LABEL[db], "method": M_LABEL[m], "top_class_share_pct": np.mean(top),
                         "classes_never_predicted_pct": np.mean(never), "normalized_entropy": np.mean(ent)})
    return pd.DataFrame(rows)


def seed_stability(data):
    rows = []
    for db in DBS:
        for m in METHODS:
            row = {"database": DB_LABEL[db], "method": M_LABEL[m]}
            vals = []
            for s in SEEDS:
                v = np.mean([data[(db, m)][s][i]["accuracy"] for i in subject_ids(data, db)]) * 100
                row[f"seed_{s}"] = v
                vals.append(v)
            row["mean"] = np.mean(vals)
            row["sd_across_seeds"] = np.std(vals, ddof=1)
            rows.append(row)
    return pd.DataFrame(rows)


def gain_correlations(data):
    rows = []
    for db in DBS:
        ids = subject_ids(data, db)
        gain = (seed_avg(data, db, "coral", "accuracy") - seed_avg(data, db, "raw", "accuracy")) * 100
        cov0 = np.array([np.mean([data[(db, "coral")][s][i]["domain_shift"]["cov_before"] for s in SEEDS]) for i in ids])
        ncal = np.array([np.mean([data[(db, "coral")][s][i]["n_calibration_samples_used"] for s in SEEDS]) for i in ids])
        raw = seed_avg(data, db, "raw", "accuracy") * 100
        for name, x in [("initial covariance distance", cov0), ("calibration windows", ncal), ("Raw accuracy", raw)]:
            rho, p = stats.spearmanr(x, gain)
            rows.append({"database": DB_LABEL[db], "predictor": name, "spearman_rho": rho, "p_value": p})
    return pd.DataFrame(rows)


def timing_table(data):
    rows = []
    for db in DBS:
        for m in METHODS:
            ad = [data[(db, m)][s][i]["timing"].get("adaptation", np.nan) for s in SEEDS for i in subject_ids(data, db)]
            ff = [data[(db, m)][s][i]["timing"].get("feature_fit", np.nan) for s in SEEDS for i in subject_ids(data, db)]
            rows.append({"database": DB_LABEL[db], "method": M_LABEL[m],
                         "adaptation_s_mean": np.nanmean(ad), "adaptation_s_sd": np.nanstd(ad, ddof=1),
                         "feature_fit_s_mean": np.nanmean(ff)})
    return pd.DataFrame(rows)


def perclass_table(data):
    rows = []
    for db in DBS:
        ids = subject_ids(data, db)
        K = n_classes(data, db)
        for m in METHODS:
            f = np.array([[np.mean([data[(db, m)][s][i]["f1_per_class"][c] for s in SEEDS]) for c in range(K)] for i in ids])
            for c in range(K):
                rows.append({"database": DB_LABEL[db], "method": M_LABEL[m], "class_index": c + 1,
                             "f1_mean_pct": f[:, c].mean() * 100})
    return pd.DataFrame(rows)


def sensitivity_table(results_dir, data):
    path = os.path.join(results_dir, "sensitivity_analysis_db7_feature_reuse.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf8") as fh:
        j = json.load(fh)
    lab = {"coral_reg": "CORAL lambda", "tca_components": "TCA components", "sa_components": "SA components"}
    rows = [{"parameter": lab.get(r["param"], r["param"]), "value": r["value"],
             "accuracy_pct": r["accuracy_mean"] * 100, "sd_across_subjects_pct": r["accuracy_std"] * 100,
             "balanced_acc_pct": r["balanced_accuracy_mean"] * 100, "macro_f1_pct": r["macro_f1_mean"] * 100,
             "n_folds": r["n_folds"]} for r in j["rows"]]
    df = pd.DataFrame(rows)
    ref = {k: np.mean([data[("db7", "raw")][42][i][k] for i in subject_ids(data, "db7")]) * 100
           for k in ("accuracy", "balanced_accuracy", "macro_f1")}
    df.attrs["reference_raw_seed42"] = ref
    return df


# --------------------------------------------------------------------------------------
# output helpers
# --------------------------------------------------------------------------------------
def save(df, name, out_dir):
    path = os.path.join(out_dir, name + ".csv")
    df.to_csv(path, index=False, float_format="%.6g")
    print(f"  [table] {path}")


# --------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------
def make_figures(data, tables, fig_dir, sens):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 100, "savefig.dpi": 300})
    col = {"raw": "#6c757d", "coral": "#d95f02", "tca": "#1b9e77", "sa": "#7570b3"}
    pair = tables["pairwise"]

    def stars(p):
        return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."

    def savefig(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(fig_dir, f"{name}.{ext}"), bbox_inches="tight")
        plt.close(fig)
        print(f"  [figure] {name}")

    # Fig 2 / Fig 3: subject-level distributions
    for fname, keys in [("figure2_accuracy_by_condition", [("accuracy", "Accuracy (%)")]),
                        ("figure3_balanced_acc_macro_f1", [("balanced_accuracy", "Balanced accuracy (%)"),
                                                           ("macro_f1", "Macro-F1 (%)")])]:
        fig, axes = plt.subplots(len(keys), 3, figsize=(10, 3.2 * len(keys)), squeeze=False)
        for r, (key, ylabel) in enumerate(keys):
            for c, db in enumerate(DBS):
                ax = axes[r][c]
                vals = [seed_avg(data, db, m, key) * 100 for m in METHODS]
                ax.boxplot(vals, widths=0.5, showfliers=False)
                rng = np.random.RandomState(1)
                for k, v in enumerate(vals):
                    ax.scatter(rng.normal(k + 1, 0.05, len(v)), v, s=9, color=col[METHODS[k]], alpha=0.8, zorder=3)
                    ax.hlines(v.mean(), k + 0.7, k + 1.3, color="k", lw=2)
                ax.axhline(100.0 / n_classes(data, db), color="red", ls="--", lw=1)
                ax.set_xticks(range(1, 5))
                ax.set_xticklabels([M_LABEL[m] for m in METHODS])
                lab = dict(METRICS)[key]
                top = max(v.max() for v in vals)
                for k, m in enumerate(METHODS[1:], start=2):
                    sel = pair[(pair.database == DB_LABEL[db]) & (pair.metric == lab) &
                               (pair.contrast == f"Raw vs {M_LABEL[m]}")].iloc[0]
                    ax.text(k, top * 1.03, stars(sel.p_holm_sidak), ha="center", fontsize=8)
                ax.set_title(f"({'abc'[c] if r == 0 else 'def'[c]}) {DB_LABEL[db]}")
                if c == 0:
                    ax.set_ylabel(ylabel)
        fig.tight_layout()
        savefig(fig, fname)

    # Fig 4: paired differences
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6))
    rng = np.random.RandomState(7)
    for r, m in enumerate(["coral", "tca"]):
        for c, db in enumerate(DBS):
            ax = axes[r][c]
            d = (seed_avg(data, db, m, "accuracy") - seed_avg(data, db, "raw", "accuracy")) * 100
            ids = np.array(subject_ids(data, db))
            o = np.argsort(d)
            colors = ["k" if (db == "db7" and ids[k] in DB7_AMPUTEES) else col[m] for k in o]
            ax.bar(range(len(d)), d[o], color=colors)
            lo, hi = boot_ci(d, rng)
            ax.axhline(d.mean(), color="k", lw=1.5)
            ax.axhspan(lo, hi, color="k", alpha=0.12)
            ax.axhline(0, color="grey", lw=0.6)
            ax.set_title(f"({'abcdef'[r * 3 + c]}) {M_LABEL[m]} - Raw, {DB_LABEL[db]}")
            ax.set_xlabel("held-out subjects (sorted)")
            if c == 0:
                ax.set_ylabel("accuracy difference (pp)")
    fig.tight_layout()
    savefig(fig, "figure4_paired_differences")

    # Fig 6: CORAL covariance distance before/after
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for c, db in enumerate(DBS):
        ax = axes[c]
        ids = subject_ids(data, db)
        b = np.array([np.mean([data[(db, "coral")][s][i]["domain_shift"]["cov_before"] for s in SEEDS]) for i in ids])
        a = np.array([np.mean([data[(db, "coral")][s][i]["domain_shift"]["cov_after"] for s in SEEDS]) for i in ids])
        for x, y in zip(b, a):
            ax.plot([0, 1], [x, y], color="grey", lw=0.6, alpha=0.7)
        ax.plot([0, 1], [b.mean(), a.mean()], color="k", lw=2.5)
        ax.set_yscale("log")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["before CORAL", "after CORAL"])
        ax.set_xlim(-0.2, 1.2)
        ax.set_title(f"({'abc'[c]}) {DB_LABEL[db]}")
        if c == 0:
            ax.set_ylabel("source-target covariance distance")
    fig.tight_layout()
    savefig(fig, "figure6_coral_covariance_shift")

    # Fig 7: prediction balance
    pb = tables["balance"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, (colname, ylabel, ttl) in zip(axes, [("top_class_share_pct", "share of predictions (%)", "(a) most-predicted class"),
                                                  ("classes_never_predicted_pct", "classes (%)", "(b) classes never predicted"),
                                                  ("normalized_entropy", "normalized entropy", "(c) entropy of predictions")]):
        w = 0.2
        for k, m in enumerate(METHODS):
            v = [pb[(pb.database == DB_LABEL[db]) & (pb.method == M_LABEL[m])][colname].iloc[0] for db in DBS]
            ax.bar(np.arange(3) + (k - 1.5) * w, v, w, color=col[m], label=M_LABEL[m])
        ax.set_xticks(range(3))
        ax.set_xticklabels([DB_LABEL[d].replace("NinaPro ", "") for d in DBS])
        ax.set_ylabel(ylabel)
        ax.set_title(ttl)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    savefig(fig, "figure7_prediction_balance")

    # Fig 8 / 9 / S1: confusion matrices Raw vs CORAL
    for db, fname in [("db7", "figure8_confusion_db7"), ("db3", "figure9_confusion_db3"), ("db2", "figureS1_confusion_db2")]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
        for ax, m in zip(axes, ["raw", "coral"]):
            K = n_classes(data, db)
            cm = np.zeros((K, K))
            for s in SEEDS:
                for i in subject_ids(data, db):
                    cm += np.asarray(data[(db, m)][s][i]["confusion_matrix"], dtype=float)
            rows_sum = cm.sum(axis=1, keepdims=True)
            cmn = np.divide(cm, rows_sum, out=np.zeros_like(cm), where=rows_sum > 0)
            im = ax.imshow(cmn, cmap="viridis", vmin=0, vmax=max(0.5, cmn.max() * 0.9))
            ax.set_title(f"{M_LABEL[m]} - {DB_LABEL[db]}")
            ax.set_xlabel("predicted class")
            ax.set_ylabel("true class")
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        savefig(fig, fname)

    # Fig 10 / 11: sensitivity
    if sens is not None:
        ref = sens.attrs["reference_raw_seed42"]
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        d = sens[sens.parameter == "CORAL lambda"].sort_values("value")
        ax.plot(d.value, d.accuracy_pct, "o-", color=col["coral"])
        ax.axhline(ref["accuracy"], color="grey", ls=":", label="Raw (seed 42)")
        ax.axvline(1e-3, color="k", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("CORAL regularization lambda")
        ax.set_ylabel("accuracy (%)")
        ax.legend(frameon=False)
        savefig(fig, "figure10_sensitivity_coral")
        fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
        for ax, (p, m, ttl) in zip(axes, [("TCA components", "tca", "(a) TCA"), ("SA components", "sa", "(b) SA")]):
            d = sens[sens.parameter == p].sort_values("value")
            ax.plot(d.value, d.accuracy_pct, "o-", color=col[m])
            ax.axhline(ref["accuracy"], color="grey", ls=":")
            ax.set_xscale("log")
            ax.set_xlabel("number of components")
            ax.set_ylabel("accuracy (%)")
            ax.set_title(ttl)
        fig.tight_layout()
        savefig(fig, "figure11_sensitivity_dimension")

    # Fig 12: seeds
    ss = tables["seeds"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for c, db in enumerate(DBS):
        ax = axes[c]
        for k, m in enumerate(METHODS):
            row = ss[(ss.database == DB_LABEL[db]) & (ss.method == M_LABEL[m])].iloc[0]
            v = [row[f"seed_{s}"] for s in SEEDS]
            ax.bar(k, row["mean"], color=col[m], alpha=0.5)
            ax.scatter([k] * len(v), v, color=col[m], s=14, zorder=3)
        ax.axhline(100.0 / n_classes(data, db), color="red", ls="--", lw=1)
        ax.set_xticks(range(4))
        ax.set_xticklabels([M_LABEL[m] for m in METHODS])
        ax.set_title(f"({'abc'[c]}) {DB_LABEL[db]}")
        if c == 0:
            ax.set_ylabel("database-mean accuracy (%)")
    fig.tight_layout()
    savefig(fig, "figure12_seed_stability")


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default=os.path.join(ROOT, "outputs", "results"))
    ap.add_argument("--tables_dir", default=os.path.join(ROOT, "outputs", "tables"))
    ap.add_argument("--figures_dir", default=os.path.join(ROOT, "outputs", "figures"))
    ap.add_argument("--no_figures", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.tables_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    print("Loading results ...")
    data = load_all(args.results_dir)
    rng = np.random.RandomState(BOOT_SEED)

    # centering check (article Section 2.3): Raw vs Centering over all method-fold evaluations
    diffs = []
    for db in DBS:
        for s in SEEDS:
            for i in subject_ids(data, db):
                diffs.append(abs(data[(db, "raw")][s][i]["accuracy"] - data[(db, "centering")][s][i]["accuracy"]) * 100)
    diffs = np.array(diffs)
    print(f"  Centering check: {len(diffs)} paired evaluations, {100 * np.mean(diffs > 1e-9):.2f}% differ, max {diffs.max():.2f} pp")

    print("Writing tables ...")
    t = {}
    t["volumes"] = table1_volumes(data)
    t["main"] = table2_main(data, rng)
    t["friedman"], t["pairwise"], t["effects"], t["restricted"] = stats_tables(data, rng)
    t["persubject"] = per_subject_table(data)
    t["shift"] = coral_shift_table(data)
    t["subgroups"] = subgroup_table(data)
    t["balance"] = prediction_balance(data)
    t["seeds"] = seed_stability(data)
    t["corr"] = gain_correlations(data)
    t["timing"] = timing_table(data)
    t["perclass"] = perclass_table(data)
    sens = sensitivity_table(args.results_dir, data)

    names = [("volumes", "Table1_data_volumes"), ("main", "Table2_main_results"),
             ("friedman", "Table3_friedman"), ("effects", "Table4_effects_vs_raw"),
             ("shift", "Table5_coral_covariance_shift"), ("subgroups", "Table6_population_subgroups"),
             ("pairwise", "TableS_pairwise_contrasts_all_metrics"),
             ("restricted", "TableS_restricted_family_three_contrasts"),
             ("persubject", "TableS_per_subject_metrics"), ("balance", "TableS_prediction_balance"),
             ("seeds", "TableS_seed_stability"), ("corr", "TableS_gain_correlations"),
             ("timing", "TableS_timing"), ("perclass", "TableS_per_class_f1")]
    for key, name in names:
        save(t[key], name, args.tables_dir)
    if sens is not None:
        save(sens, "Table7_sensitivity_db7_seed42", args.tables_dir)

    if not args.no_figures:
        print("Writing figures ...")
        make_figures(data, t, args.figures_dir, sens)
    print("Done.")


if __name__ == "__main__":
    main()
