from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import save_yaml
from src.utils.config_pack import load_yaml_with_pack
from src.utils.io import read_json, write_json
from src.utils.resource import get_git_commit


def _set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    keys = [k for k in str(dotted).split(".") if k]
    cur = cfg
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def _apply_params(cfg: dict, params: dict[str, Any]) -> dict:
    out = copy.deepcopy(cfg)
    for k, v in params.items():
        _set_dotted(out, k, v)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply tuned hyperparameters to configs/tuned")
    parser.add_argument("--best", type=str, default="./artifacts/tuning/best_params.json")
    parser.add_argument("--out_dir", type=str, default="./configs/tuned")
    parser.add_argument("--base_pack", type=str, default="base")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    best = read_json(args.best)
    out_dir = Path(args.out_dir)
    if out_dir.exists() and not args.overwrite:
        # Non-destructive default; users can force overwrite explicitly.
        pass

    eegnet_base = load_yaml_with_pack("configs/models/eegnet.yaml", config_pack=args.base_pack)
    gen_base = load_yaml_with_pack("configs/generators/cwgan_gp.yaml", config_pack=args.base_pack)
    qc_base = load_yaml_with_pack("configs/qc.yaml", config_pack=args.base_pack)

    eegnet_params = best.get("eegnet", {}).get("params", {})
    gen_params = best.get("genaug_qc", {}).get("params", {}).get("generator", {})
    qc_params = best.get("genaug_qc", {}).get("params", {}).get("qc", {})

    eegnet_tuned = _apply_params(eegnet_base, eegnet_params)
    gen_tuned = _apply_params(gen_base, gen_params)
    qc_tuned = _apply_params(qc_base, qc_params)

    save_yaml(out_dir / "models" / "eegnet.yaml", eegnet_tuned)
    save_yaml(out_dir / "generators" / "cwgan_gp.yaml", gen_tuned)
    save_yaml(out_dir / "qc.yaml", qc_tuned)

    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": get_git_commit(),
        "best_source": str(Path(args.best).resolve()),
        "base_pack": str(args.base_pack),
        "written": [
            str((out_dir / "models" / "eegnet.yaml").as_posix()),
            str((out_dir / "generators" / "cwgan_gp.yaml").as_posix()),
            str((out_dir / "qc.yaml").as_posix()),
        ],
    }
    write_json(out_dir / "manifest.json", manifest)
    print(f"[tuned] wrote configs under {out_dir}")


if __name__ == "__main__":
    main()
