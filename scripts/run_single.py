from __future__ import annotations

import argparse

from src.train.pipeline import run_experiment
from src.utils.config import load_yaml
from src.utils.results import append_result, has_primary_key, load_results


def _load_all_configs(overrides: list[str]) -> dict:
    cfg = {
        "dataset": load_yaml("configs/dataset_bci2a.yaml", overrides=overrides),
        "preprocess": load_yaml("configs/preprocess.yaml", overrides=overrides),
        "split": load_yaml("configs/split.yaml", overrides=overrides),
        "qc": load_yaml("configs/qc.yaml", overrides=overrides),
        "models": {
            "eegnet": load_yaml("configs/models/eegnet.yaml", overrides=overrides),
            "eegconformer": load_yaml("configs/models/eegconformer.yaml", overrides=overrides),
            "ctnet": load_yaml("configs/models/ctnet.yaml", overrides=overrides),
            "svm": load_yaml("configs/models/fbcsp_svm.yaml", overrides=overrides),
        },
        "generators": {
            "cwgan_gp": load_yaml("configs/generators/cwgan_gp.yaml", overrides=overrides),
            "ddpm": load_yaml("configs/generators/ddpm.yaml", overrides=overrides),
        },
    }
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single experiment")
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--r", type=float, required=True)
    parser.add_argument("--method", type=str, required=True, choices=["C0", "C1", "C2", "GenAug"])
    parser.add_argument("--classifier", type=str, required=True, choices=["eegnet", "eegconformer", "ctnet", "svm"])
    parser.add_argument("--generator", type=str, default="cwgan_gp", choices=["cwgan_gp", "ddpm"])
    parser.add_argument("--alpha_ratio", type=float, default=0.0)
    parser.add_argument("--qc_on", action="store_true")
    parser.add_argument("--stage", type=str, default="full", choices=["alpha_search", "final_eval", "full"])
    parser.add_argument("--results", type=str, default="results/results.csv")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = _load_all_configs(args.override)
    method = args.method
    generator = args.generator if method == "GenAug" else "none"
    alpha_ratio = float(args.alpha_ratio) if method == "GenAug" else 0.0
    qc_on = bool(args.qc_on) if method == "GenAug" else False

    row_key = {
        "subject": args.subject,
        "seed": args.seed,
        "r": float(args.r),
        "classifier": args.classifier,
        "method": method,
        "generator": generator,
        "qc_on": bool(qc_on),
        "alpha_ratio": float(alpha_ratio),
    }
    df = load_results(args.results)
    if has_primary_key(df, row_key):
        return

    row = run_experiment(
        subject=args.subject,
        seed=args.seed,
        r=float(args.r),
        method=method,
        classifier=args.classifier,
        generator=generator,
        alpha_ratio=alpha_ratio,
        qc_on=qc_on,
        dataset_cfg=cfg["dataset"],
        preprocess_cfg=cfg["preprocess"],
        split_cfg=cfg["split"],
        model_cfgs=cfg["models"],
        gen_cfgs=cfg["generators"],
        qc_cfg=cfg["qc"],
        results_path=args.results,
        stage=args.stage,
        compute_distance=True,
    )
    append_result(args.results, row)


if __name__ == "__main__":
    main()
