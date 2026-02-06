from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.pipeline import run_experiment
from src.utils.config import load_yaml
from src.utils.config_pack import load_yaml_with_pack
from src.utils.results import append_result, has_primary_key, load_results


def _load_all_configs(overrides: list[str], config_pack: str = "base") -> dict:
    """Load all required YAML configs with optional overrides.

    Inputs:
    - overrides: list of key=value override strings.

    Outputs:
    - dict containing dataset/preprocess/split/qc/models/generators configs.

    Internal logic:
    - Uses load_yaml for each config file with the same overrides list.
    """
    cfg = {
        "dataset": load_yaml_with_pack("configs/dataset_bci2a.yaml", config_pack=config_pack, overrides=overrides),
        "preprocess": load_yaml_with_pack("configs/preprocess.yaml", config_pack=config_pack, overrides=overrides),
        "split": load_yaml_with_pack("configs/split.yaml", config_pack=config_pack, overrides=overrides),
        "qc": load_yaml_with_pack("configs/qc.yaml", config_pack=config_pack, overrides=overrides),
        "models": {
            "eegnet": load_yaml_with_pack("configs/models/eegnet.yaml", config_pack=config_pack, overrides=overrides),
            "eegconformer": load_yaml_with_pack("configs/models/eegconformer.yaml", config_pack=config_pack, overrides=overrides),
            "ctnet": load_yaml_with_pack("configs/models/ctnet.yaml", config_pack=config_pack, overrides=overrides),
            "svm": load_yaml_with_pack("configs/models/fbcsp_svm.yaml", config_pack=config_pack, overrides=overrides),
        },
        "generators": {
            "cwgan_gp": load_yaml_with_pack("configs/generators/cwgan_gp.yaml", config_pack=config_pack, overrides=overrides),
            "ddpm": load_yaml_with_pack("configs/generators/ddpm.yaml", config_pack=config_pack, overrides=overrides),
        },
        "config_pack": str(config_pack),
    }
    return cfg


def main() -> None:
    """Run a single experiment configuration and append results.

    Inputs:
    - CLI args define subject/seed/r/method/classifier/alpha/qc_on/stage.

    Outputs:
    - Appends a single row to results.csv and writes run artifacts.

    Internal logic:
    - Loads configs, builds a primary key, skips if already present, otherwise
      runs the full pipeline for the given configuration.
    """
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
    parser.add_argument("--config_pack", type=str, default="base")
    parser.add_argument("--results", type=str, default="results/results.csv")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = _load_all_configs(args.override, config_pack=args.config_pack)  # all YAML configs
    method = args.method  # C0/C1/C2/GenAug
    generator = args.generator if method == "GenAug" else "none"  # generator type or none
    alpha_ratio = float(args.alpha_ratio) if method == "GenAug" else 0.0  # synth:real ratio
    qc_on = bool(args.qc_on) if method == "GenAug" else False  # QC flag only for GenAug

    row_key = {  # primary key for resume-safe execution
        "config_pack": str(args.config_pack),
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

    exp_cfg = load_yaml("configs/experiment_grid.yaml", overrides=args.override)
    alpha_search_cfg = exp_cfg.get("alpha_search", {})
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
        alpha_search_cfg=alpha_search_cfg,
        config_pack=str(args.config_pack),
    )
    append_result(args.results, row)


if __name__ == "__main__":
    main()
