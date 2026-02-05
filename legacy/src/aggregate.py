from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr, rankdata, ttest_rel, wilcoxon

METRICS = ["acc", "kappa", "f1_macro"]


def _present_metrics(df: pd.DataFrame) -> list[str]:
    return [m for m in METRICS if m in df.columns]


def load_per_run_metrics(metrics_dir: str | Path, metrics_filename: str = "clf_cross_session.csv") -> pd.DataFrame:
    metrics_dir = Path(metrics_dir)
    primary = metrics_dir / metrics_filename
    if primary.exists():
        return pd.read_csv(primary)

    files = sorted(metrics_dir.glob("clf_*.csv"))
    if not files:
        raise RuntimeError(f"No metric files found under {metrics_dir}")

    dfs = []
    required = {"acc", "kappa", "f1_macro", "clf_model"}
    for f in files:
        df = pd.read_csv(f)
        if required.issubset(set(df.columns)):
            dfs.append(df)
    if not dfs:
        raise RuntimeError(f"No compatible clf metric files found under {metrics_dir}")
    return pd.concat(dfs, ignore_index=True)


def aggregate_seed_mean_std(df: pd.DataFrame) -> pd.DataFrame:
    metrics = _present_metrics(df)
    if not metrics:
        raise RuntimeError("No metric columns found for aggregation.")

    group_cols = [
        "protocol",
        "subject",
        "p",
        "clf_model",
        "condition",
        "gen_model",
        "mode",
        "ratio",
        "alpha_tilde",
        "qc_on",
    ]
    group_cols = [c for c in group_cols if c in df.columns]

    agg = df.groupby(group_cols, dropna=False)[metrics].agg(["mean", "std", "count"]).reset_index()
    agg.columns = ["_".join(c).strip("_") for c in agg.columns]

    for m in metrics:
        cnt_col = f"{m}_count"
        std_col = f"{m}_std"
        ci_col = f"{m}_ci95"
        agg[ci_col] = 1.96 * agg[std_col] / np.sqrt(np.maximum(agg[cnt_col], 1))
    return agg


def aggregate_over_subjects(seed_agg: pd.DataFrame) -> pd.DataFrame:
    metrics = [m for m in METRICS if f"{m}_mean" in seed_agg.columns]
    if not metrics:
        raise RuntimeError("No seed-level metric columns found for subject aggregation.")

    base_cols = [
        "protocol",
        "p",
        "clf_model",
        "condition",
        "gen_model",
        "mode",
        "ratio",
        "alpha_tilde",
        "qc_on",
    ]
    base_cols = [c for c in base_cols if c in seed_agg.columns]

    use_cols = [c for c in seed_agg.columns if c.endswith("_mean") and c.split("_mean")[0] in METRICS]
    grp = seed_agg.groupby(base_cols, dropna=False)[use_cols].agg(["mean", "std", "count"]).reset_index()
    grp.columns = ["_".join(c).strip("_") for c in grp.columns]

    out = grp.copy()
    for m in metrics:
        mean_mean = f"{m}_mean_mean"
        mean_std = f"{m}_mean_std"
        mean_cnt = f"{m}_mean_count"
        out[f"{m}_mean"] = out[mean_mean]
        out[f"{m}_std"] = out[mean_std]
        out[f"{m}_ci95"] = 1.96 * out[mean_std] / np.sqrt(np.maximum(out[mean_cnt], 1))
    return out


def reproducibility_check(df: pd.DataFrame, min_seed_runs: int = 3) -> pd.DataFrame:
    metrics = _present_metrics(df)
    base = df[df["condition"] == "C0_no_aug"].copy() if "condition" in df.columns else df.copy()

    def _is_finite(s: pd.Series) -> bool:
        return bool(np.isfinite(s.astype(float)).all())

    rows = []
    for clf, sub in base.groupby("clf_model"):
        n_runs = int(len(sub))
        n_seeds = int(sub["seed"].nunique()) if "seed" in sub.columns else 0
        finite_ok = all(_is_finite(sub[m]) for m in metrics if m in sub.columns)

        rows.append(
            {
                "clf_model": clf,
                "n_runs": n_runs,
                "n_unique_seeds": n_seeds,
                "acc_mean": float(sub["acc"].mean()) if "acc" in sub.columns else math.nan,
                "acc_std": float(sub["acc"].std()) if "acc" in sub.columns else math.nan,
                "kappa_mean": float(sub["kappa"].mean()) if "kappa" in sub.columns else math.nan,
                "f1_macro_mean": float(sub["f1_macro"].mean()) if "f1_macro" in sub.columns else math.nan,
                "finite_ok": finite_ok,
                "repro_pass": bool(finite_ok and n_seeds >= min_seed_runs),
            }
        )

    return pd.DataFrame(rows).sort_values(["repro_pass", "clf_model"], ascending=[False, True])


def add_baseline_gain(df: pd.DataFrame) -> pd.DataFrame:
    metrics = _present_metrics(df)
    key_cols = ["split", "subject", "seed", "p", "clf_model"]
    key_cols = [c for c in key_cols if c in df.columns]

    base = df[df["condition"] == "C0_no_aug"][key_cols + metrics].copy()
    base = base.rename(columns={m: f"baseline_{m}" for m in metrics})

    out = df.merge(base, on=key_cols, how="left")
    for m in metrics:
        out[f"gain_{m}"] = out[m] - out[f"baseline_{m}"]
    return out


def _rank_biserial(delta: np.ndarray) -> float:
    nonzero = delta[np.abs(delta) > 1e-12]
    if len(nonzero) == 0:
        return float("nan")
    ranks = rankdata(np.abs(nonzero))
    w_pos = float(ranks[nonzero > 0].sum())
    w_neg = float(ranks[nonzero < 0].sum())
    denom = w_pos + w_neg
    if denom <= 0:
        return float("nan")
    return (w_pos - w_neg) / denom


def paired_condition_stats(df: pd.DataFrame) -> pd.DataFrame:
    metrics = _present_metrics(df)
    if "condition" not in df.columns or "subject" not in df.columns:
        return pd.DataFrame()

    rows = []
    key_cols = ["protocol", "p", "clf_model"]
    key_cols = [c for c in key_cols if c in df.columns]
    cond_cols = ["condition", "gen_model", "mode", "ratio", "alpha_tilde", "qc_on", "evaluated_on"]
    cond_cols = [c for c in cond_cols if c in df.columns]

    base = df[df["condition"] == "C0_no_aug"].copy()
    cond_df = df[df["condition"] != "C0_no_aug"].copy()
    if len(base) == 0 or len(cond_df) == 0:
        return pd.DataFrame()

    for group_vals, grp in cond_df.groupby(key_cols + cond_cols, dropna=False):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        gmap = dict(zip(key_cols + cond_cols, group_vals))

        base_sub = base.copy()
        for k in key_cols:
            base_sub = base_sub[base_sub[k] == gmap[k]]
        if len(base_sub) == 0:
            continue

        for m in metrics:
            cond_s = grp.groupby("subject", dropna=False)[m].mean()
            base_s = base_sub.groupby("subject", dropna=False)[m].mean()
            paired = pd.concat([base_s.rename("baseline"), cond_s.rename("condition")], axis=1, join="inner").dropna()
            if len(paired) == 0:
                continue

            delta = (paired["condition"] - paired["baseline"]).to_numpy(dtype=float)
            n = int(len(delta))
            mean_delta = float(np.mean(delta))
            median_delta = float(np.median(delta))
            std_delta = float(np.std(delta, ddof=1)) if n > 1 else float("nan")
            ci_half = 1.96 * std_delta / np.sqrt(n) if n > 1 and np.isfinite(std_delta) else float("nan")

            p_t = float("nan")
            p_w = float("nan")
            if n >= 2:
                try:
                    p_t = float(ttest_rel(paired["condition"], paired["baseline"], nan_policy="omit").pvalue)
                except Exception:
                    p_t = float("nan")
                try:
                    p_w = float(wilcoxon(delta, zero_method="wilcox", alternative="two-sided").pvalue)
                except Exception:
                    p_w = float("nan")

            effect_d = float(mean_delta / std_delta) if n > 1 and std_delta > 0 else float("nan")
            effect_rbc = _rank_biserial(delta)

            row = {
                **gmap,
                "metric": m,
                "n_subjects": n,
                "baseline_mean": float(np.mean(paired["baseline"])),
                "condition_mean": float(np.mean(paired["condition"])),
                "mean_delta": mean_delta,
                "median_delta": median_delta,
                "delta_std": std_delta,
                "ci95_low": float(mean_delta - ci_half) if np.isfinite(ci_half) else float("nan"),
                "ci95_high": float(mean_delta + ci_half) if np.isfinite(ci_half) else float("nan"),
                "p_value_ttest": p_t,
                "p_value_wilcoxon": p_w,
                "effect_size_cohen_d_paired": effect_d,
                "effect_size_rank_biserial": effect_rbc,
            }
            rows.append(row)

    return pd.DataFrame(rows)


def distance_gain_correlation(df: pd.DataFrame) -> pd.DataFrame:
    if "dist_swd" not in df.columns and "dist_mmd" not in df.columns:
        return pd.DataFrame()

    rows = []
    cand = df[(df.get("ratio", 0) > 0)].copy()

    for dist_col in ["dist_swd", "dist_mmd"]:
        if dist_col not in cand.columns:
            continue

        use = cand[np.isfinite(cand[dist_col]) & np.isfinite(cand["gain_acc"])]
        if len(use) >= 3 and float(use[dist_col].std()) > 0 and float(use["gain_acc"].std()) > 0:
            corr, pval = pearsonr(use[dist_col], use["gain_acc"])
            reg = linregress(use[dist_col], use["gain_acc"])
            rows.append(
                {
                    "scope": "overall",
                    "distance": dist_col,
                    "n": int(len(use)),
                    "pearson_r": float(corr),
                    "pearson_p": float(pval),
                    "slope": float(reg.slope),
                    "intercept": float(reg.intercept),
                    "r2": float(reg.rvalue**2),
                    "reg_p": float(reg.pvalue),
                }
            )

        for clf, sub in use.groupby("clf_model"):
            if len(sub) < 3 or float(sub[dist_col].std()) == 0 or float(sub["gain_acc"].std()) == 0:
                continue
            corr, pval = pearsonr(sub[dist_col], sub["gain_acc"])
            reg = linregress(sub[dist_col], sub["gain_acc"])
            rows.append(
                {
                    "scope": f"clf:{clf}",
                    "distance": dist_col,
                    "n": int(len(sub)),
                    "pearson_r": float(corr),
                    "pearson_p": float(pval),
                    "slope": float(reg.slope),
                    "intercept": float(reg.intercept),
                    "r2": float(reg.rvalue**2),
                    "reg_p": float(reg.pvalue),
                }
            )

    return pd.DataFrame(rows)


def _plot_accuracy_vs_ratio(df: pd.DataFrame, out_path: Path) -> None:
    if "ratio" not in df.columns:
        return

    plot_df = (
        df.groupby(["clf_model", "condition", "ratio"], dropna=False)["acc"]
        .mean()
        .reset_index()
        .sort_values(["clf_model", "condition", "ratio"])
    )

    clf_list = sorted(plot_df["clf_model"].unique().tolist())
    n = max(1, len(clf_list))
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False)

    for i, clf in enumerate(clf_list):
        ax = axes[i // ncols][i % ncols]
        sub = plot_df[plot_df["clf_model"] == clf]
        for cond, grp in sub.groupby("condition"):
            ax.plot(grp["ratio"], grp["acc"], marker="o", label=str(cond))
        ax.set_title(f"{clf}: Accuracy vs ratio (rho)")
        ax.set_xlabel("rho = N_synth / N_real")
        ax.set_ylabel("Accuracy")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    for j in range(len(clf_list), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_accuracy_vs_r(df: pd.DataFrame, out_path: Path, ratio_ref: float = 1.0) -> None:
    if "p" not in df.columns:
        return

    sub = df[(df["condition"] == "C0_no_aug") | (np.isclose(df.get("ratio", 0.0), ratio_ref))].copy()
    if len(sub) == 0:
        return

    plot_df = (
        sub.groupby(["clf_model", "condition", "p"], dropna=False)["acc"]
        .mean()
        .reset_index()
        .sort_values(["clf_model", "condition", "p"])
    )

    clf_list = sorted(plot_df["clf_model"].unique().tolist())
    n = max(1, len(clf_list))
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False)

    for i, clf in enumerate(clf_list):
        ax = axes[i // ncols][i % ncols]
        cdf = plot_df[plot_df["clf_model"] == clf]
        for cond, grp in cdf.groupby("condition"):
            ax.plot(grp["p"], grp["acc"], marker="o", label=str(cond))
        ax.set_title(f"{clf}: Accuracy vs low-data ratio r")
        ax.set_xlabel("r (low-data fraction)")
        ax.set_ylabel("Accuracy")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    for j in range(len(clf_list), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_distance_vs_gain(df: pd.DataFrame, out_path: Path, dist_col: str = "dist_swd") -> None:
    if dist_col not in df.columns or "gain_acc" not in df.columns:
        return

    sub = df[np.isfinite(df[dist_col]) & np.isfinite(df["gain_acc"]) & (df.get("ratio", 0.0) > 0)].copy()
    if len(sub) < 3:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for cond, grp in sub.groupby("condition"):
        ax.scatter(grp[dist_col], grp["gain_acc"], s=20, alpha=0.7, label=str(cond))

    reg = linregress(sub[dist_col], sub["gain_acc"])
    xs = np.linspace(float(sub[dist_col].min()), float(sub[dist_col].max()), 100)
    ys = reg.intercept + reg.slope * xs
    ax.plot(xs, ys, "k--", linewidth=1.5, label=f"fit (R^2={reg.rvalue**2:.3f})")

    ax.set_xlabel(dist_col)
    ax.set_ylabel("Accuracy gain vs baseline")
    ax.set_title("Distance vs Accuracy Gain")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def aggregate_metrics(
    metrics_dir: str | Path,
    out_csv: str | Path,
    ratio_ref_for_r_curve: float = 1.0,
    metrics_filename: str = "clf_cross_session.csv",
) -> dict[str, pd.DataFrame]:
    per_run = load_per_run_metrics(metrics_dir, metrics_filename=metrics_filename)
    if "ratio" not in per_run.columns and "aug_strength" in per_run.columns:
        per_run["ratio"] = per_run["aug_strength"].astype(float)
    if "alpha_tilde" not in per_run.columns and "ratio" in per_run.columns:
        per_run["alpha_tilde"] = per_run["ratio"].astype(float) / (1.0 + per_run["ratio"].astype(float))

    seed_agg = aggregate_seed_mean_std(per_run)
    subj_agg = aggregate_over_subjects(seed_agg)
    repro = reproducibility_check(per_run)
    passed_models = set(repro[repro["repro_pass"]]["clf_model"].tolist())

    with_gain = add_baseline_gain(per_run)
    corr = distance_gain_correlation(with_gain)
    stats = paired_condition_stats(per_run)

    out_csv = Path(out_csv)
    tables_dir = out_csv.parent
    figures_dir = tables_dir.parent / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    per_run.to_csv(tables_dir / "per_run.csv", index=False)
    seed_agg.to_csv(tables_dir / "seed_aggregate.csv", index=False)
    subj_agg.to_csv(tables_dir / "subject_aggregate.csv", index=False)
    subj_agg[subj_agg["clf_model"].isin(passed_models)].to_csv(tables_dir / "subject_aggregate_passed.csv", index=False)
    repro.to_csv(tables_dir / "reproducibility_check.csv", index=False)
    with_gain.to_csv(tables_dir / "per_run_with_gain.csv", index=False)
    if len(corr) > 0:
        corr.to_csv(tables_dir / "distance_gain_correlation.csv", index=False)
    if len(stats) > 0:
        stats.to_csv(tables_dir / "stats_summary.csv", index=False)

    metrics = [m for m in METRICS if f"{m}_mean" in subj_agg.columns]
    for m in metrics:
        cols = [
            c
            for c in ["protocol", "p", "clf_model", "condition", "gen_model", "mode", "ratio", "alpha_tilde", "qc_on", f"{m}_mean", f"{m}_std", f"{m}_ci95"]
            if c in subj_agg.columns
        ]
        metric_table = subj_agg[cols].copy()
        metric_table.to_csv(tables_dir / f"main_table_{m}.csv", index=False)
        metric_table[metric_table["clf_model"].isin(passed_models)].to_csv(tables_dir / f"main_table_{m}_passed.csv", index=False)

    # Backward-compatible default output.
    subj_agg.to_csv(out_csv, index=False)

    _plot_accuracy_vs_ratio(per_run, figures_dir / "accuracy_vs_ratio.png")
    # Backward-compatible filename used in previous runs/docs.
    _plot_accuracy_vs_ratio(per_run, figures_dir / "accuracy_vs_alpha.png")
    _plot_accuracy_vs_r(per_run, figures_dir / "accuracy_vs_r.png", ratio_ref=float(ratio_ref_for_r_curve))
    _plot_distance_vs_gain(with_gain, figures_dir / "distance_vs_gain.png", dist_col="dist_swd")

    return {
        "per_run": per_run,
        "seed_agg": seed_agg,
        "subject_agg": subj_agg,
        "repro": repro,
        "corr": corr,
        "stats": stats,
    }
