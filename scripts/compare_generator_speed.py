from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Aggregate runtime statistics by group."""
    grouped = df.groupby(group_cols)["runtime_sec"]
    out = grouped.agg(
        n="count",
        mean_runtime_sec="mean",
        median_runtime_sec="median",
        std_runtime_sec="std",
    ).reset_index()
    out["p90_runtime_sec"] = grouped.quantile(0.90).to_numpy()
    return out.sort_values(group_cols).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare generator runtime from results.csv")
    parser.add_argument("--results", type=str, default="results/results.csv")
    parser.add_argument("--out", type=str, default="results/generator_speed.csv")
    parser.add_argument("--classifier", type=str, default="eegnet")
    parser.add_argument("--method", type=str, default="GenAug")
    parser.add_argument("--include_by_r", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.results)
    need_cols = {"method", "classifier", "generator", "runtime_sec"}
    missing = sorted(need_cols - set(df.columns))
    if missing:
        raise KeyError(f"Missing columns in results: {missing}")

    sub = df[
        (df["method"] == str(args.method))
        & (df["classifier"] == str(args.classifier))
        & df["runtime_sec"].notna()
        & (df["generator"] != "none")
    ].copy()
    if sub.empty:
        raise RuntimeError("No matching rows found for speed comparison.")

    for col in ("r", "alpha_ratio", "qc_on"):
        if col in sub.columns:
            sub[col] = sub[col].astype(str)

    overall = _summary(sub, ["generator"])
    if args.include_by_r and "r" in sub.columns:
        by_r = _summary(sub, ["generator", "r"])
    else:
        by_r = pd.DataFrame()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if by_r.empty:
        overall.to_csv(out_path, index=False)
    else:
        # Multi-section CSV: overall first, blank line, then by-r table.
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("# overall\n")
            overall.to_csv(f, index=False)
            f.write("\n# by_r\n")
            by_r.to_csv(f, index=False)

    print("[speed] overall")
    print(overall.to_string(index=False))
    if not by_r.empty:
        print("\n[speed] by_r")
        print(by_r.to_string(index=False))
    print(f"\n[speed] wrote {out_path}")


if __name__ == "__main__":
    main()
