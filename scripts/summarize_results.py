from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _paired_stats(diff: np.ndarray) -> tuple[float, float, float]:
    """Compute Wilcoxon p, paired t-test p, and Cohen's dz for paired differences."""
    diff = diff[np.isfinite(diff)]
    n = len(diff)
    if n < 2:
        return np.nan, np.nan, np.nan

    try:
        _, p_w = stats.wilcoxon(diff)
    except Exception:
        p_w = np.nan

    try:
        _, p_t = stats.ttest_1samp(diff, popmean=0.0)
    except Exception:
        p_t = np.nan

    sd = float(np.std(diff, ddof=1))
    if sd <= 1e-12:
        dz = np.nan
    else:
        dz = float(np.mean(diff) / sd)
    return float(p_w), float(p_t), dz


def _distance_col(df: pd.DataFrame) -> str | None:
    """Resolve the preferred distance column available in results."""
    for cand in ["distance", "dist_mmd", "dist_swd"]:
        if cand in df.columns:
            return cand
    return None


def main() -> None:
    """Aggregate results.csv into mean/std summaries.

    Inputs:
    - results.csv path and output paths.

    Outputs:
    - summary.csv: mean/std grouped by method/classifier/r/alpha/qc/generator.
    - paired_stats.csv: subject-seed paired GenAug vs C0 tests/effect sizes.
    - gain_distance.csv: gain vs distance correlation/regression diagnostics.

    Internal logic:
    - Computes grouped summaries, paired tests, and gain-distance links.
    """
    parser = argparse.ArgumentParser(description="Summarize results.csv")
    parser.add_argument("--results", type=str, default="results/results.csv")
    parser.add_argument("--out", type=str, default="results/summary.csv")
    parser.add_argument("--paired_out", type=str, default="results/paired_stats.csv")
    parser.add_argument("--gain_out", type=str, default="results/gain_distance.csv")
    parser.add_argument("--metric", type=str, default="kappa", choices=["acc", "kappa", "macro_f1"])
    args = parser.parse_args()

    df = pd.read_csv(args.results)  # full results table
    if df.empty:
        raise RuntimeError("results.csv is empty.")

    group_cols = ["method", "classifier", "r", "alpha_ratio", "qc_on", "generator"]  # grouping keys
    metrics = [m for m in ["acc", "kappa", "macro_f1"] if m in df.columns]

    summary = df.groupby(group_cols)[metrics].agg(["mean", "std"]).reset_index()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)

    # Paired GenAug vs C0 (same subject/seed/r/classifier)
    key_cols = ["subject", "seed", "r", "classifier"]
    base_cols = key_cols + metrics
    gen_cols = key_cols + ["generator", "qc_on", "alpha_ratio", "pass_rate"] + metrics
    if "distance" in df.columns:
        gen_cols.append("distance")
    if "dist_mmd" in df.columns:
        gen_cols.append("dist_mmd")

    c0 = df[df["method"] == "C0"][base_cols].copy()
    gen = df[df["method"] == "GenAug"][gen_cols].copy()
    merged = gen.merge(c0, on=key_cols, how="inner", suffixes=("_gen", "_c0"))

    paired_rows: list[dict] = []
    group_keys = ["classifier", "r", "generator", "qc_on", "alpha_ratio"]
    for key, g in merged.groupby(group_keys):
        classifier, r, generator, qc_on, alpha_ratio = key
        row = {
            "classifier": classifier,
            "r": float(r),
            "generator": generator,
            "qc_on": bool(qc_on),
            "alpha_ratio": float(alpha_ratio),
            "n_pairs": int(len(g)),
        }
        for m in metrics:
            diff = (g[f"{m}_gen"] - g[f"{m}_c0"]).to_numpy(dtype=np.float64)
            diff = diff[np.isfinite(diff)]
            p_w, p_t, dz = _paired_stats(diff)
            row[f"gain_mean_{m}"] = float(np.mean(diff)) if len(diff) else np.nan
            row[f"gain_std_{m}"] = float(np.std(diff, ddof=1)) if len(diff) > 1 else np.nan
            row[f"wilcoxon_p_{m}"] = p_w
            row[f"ttest_p_{m}"] = p_t
            row[f"effect_dz_{m}"] = dz
        paired_rows.append(row)

    paired_df = pd.DataFrame(paired_rows)
    paired_path = Path(args.paired_out)
    paired_path.parent.mkdir(parents=True, exist_ok=True)
    paired_df.to_csv(paired_path, index=False)

    # Gain vs distance diagnostics (H3)
    dist_col = _distance_col(merged)
    gain_rows: list[dict] = []
    metric = args.metric
    if dist_col is not None and f"{metric}_gen" in merged.columns and f"{metric}_c0" in merged.columns:
        g2 = merged.copy()
        g2["gain"] = g2[f"{metric}_gen"] - g2[f"{metric}_c0"]
        valid = g2[np.isfinite(g2["gain"]) & np.isfinite(g2[dist_col])]
        if not valid.empty:
            rho, p_rho = stats.spearmanr(valid["gain"], valid[dist_col])
            r_val, p_r = stats.pearsonr(valid["gain"], valid[dist_col])
            gain_rows.append(
                {
                    "metric": metric,
                    "distance_col": dist_col,
                    "n": int(len(valid)),
                    "spearman_rho": float(rho),
                    "spearman_p": float(p_rho),
                    "pearson_r": float(r_val),
                    "pearson_p": float(p_r),
                }
            )

            cols = [dist_col, "pass_rate", "alpha_ratio", "r"]
            for c in cols:
                if c not in valid.columns:
                    valid[c] = np.nan
            reg_df = valid.dropna(subset=["gain", dist_col, "pass_rate", "alpha_ratio", "r"]).copy()
            if len(reg_df) >= 5:
                x = reg_df[[dist_col, "pass_rate", "alpha_ratio", "r"]].to_numpy(dtype=np.float64)
                y = reg_df["gain"].to_numpy(dtype=np.float64)
                x = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
                coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
                y_hat = x @ coef
                ss_res = float(np.sum((y - y_hat) ** 2))
                ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                r2 = np.nan if ss_tot <= 1e-12 else float(1.0 - ss_res / ss_tot)
                gain_rows.append(
                    {
                        "metric": metric,
                        "distance_col": dist_col,
                        "n": int(len(reg_df)),
                        "reg_intercept": float(coef[0]),
                        "reg_coef_distance": float(coef[1]),
                        "reg_coef_pass_rate": float(coef[2]),
                        "reg_coef_alpha_ratio": float(coef[3]),
                        "reg_coef_r": float(coef[4]),
                        "reg_r2": r2,
                    }
                )

    gain_df = pd.DataFrame(gain_rows)
    gain_path = Path(args.gain_out)
    gain_path.parent.mkdir(parents=True, exist_ok=True)
    gain_df.to_csv(gain_path, index=False)


if __name__ == "__main__":
    main()
