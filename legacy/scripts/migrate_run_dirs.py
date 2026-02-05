#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from scripts._script_utils import project_root
except ImportError:  # pragma: no cover - direct script execution
    from _script_utils import project_root

from src.utils import load_yaml, make_exp_id, parse_p_tag


def _parse_segments(name: str) -> tuple[str, list[tuple[str, str]], list[str]]:
    parts = name.split("__")
    prefix = parts[0]
    kv: list[tuple[str, str]] = []
    extras: list[str] = []
    for seg in parts[1:]:
        if "-" in seg:
            k, v = seg.split("-", 1)
            kv.append((k, v))
        else:
            extras.append(seg)
    return prefix, kv, extras


def _kv_get(kv: list[tuple[str, str]], key: str) -> str | None:
    for k, v in kv:
        if k == key:
            return v
    return None


def _build_paper_track_maps(root: Path) -> tuple[dict[str, str], dict[str, str], dict[str, float]]:
    cfg_path = root / "configs" / "paper_track.yaml"
    if not cfg_path.exists():
        return {}, {}, {}
    cfg = load_yaml(cfg_path)
    track_map = {str(k): str(v) for k, v in cfg.get("model_track_map", {}).items()}
    aug_map = {str(k): str(v) for k, v in cfg.get("model_aug_mode", {}).items()}
    val_map = {str(k): float(v) for k, v in cfg.get("model_val_ratio", {}).items()}
    return track_map, aug_map, val_map


def _rename_gen(name: str) -> str | None:
    if "__gmodel-" in name:
        return name.replace("__gmodel-", "__gen-")
    return None


def _rename_clf(name: str, root: Path, track_map: dict[str, str], aug_map: dict[str, str], val_map: dict[str, float]) -> str | None:
    prefix, kv, extras = _parse_segments(name)

    if prefix in {"step1_none", "paper_none"}:
        subject = _kv_get(kv, "subject") or _kv_get(kv, "subj")
        seed = _kv_get(kv, "seed")
        p_val = _kv_get(kv, "p")
        clf = _kv_get(kv, "clf")
        if subject is None or seed is None or p_val is None or clf is None:
            return None
        return make_exp_id(
            prefix,
            subject=int(subject),
            seed=int(seed),
            p=parse_p_tag(p_val),
            clf=clf,
        )

    if prefix == "paper_track":
        subject = _kv_get(kv, "subject") or _kv_get(kv, "subj")
        seed = _kv_get(kv, "seed")
        clf = _kv_get(kv, "clf") or _kv_get(kv, "model")
        if subject is None or seed is None or clf is None:
            return None
        track = track_map.get(clf, "unknown")
        aug = aug_map.get(clf, "unknown")
        val = val_map.get(clf, "unknown")
        return make_exp_id(
            "paper_track",
            subject=int(subject),
            seed=int(seed),
            clf=clf,
            track=track,
            aug=aug,
            val=val,
        )

    if prefix == "official_faithful":
        subject = _kv_get(kv, "subject") or _kv_get(kv, "subj")
        seed = _kv_get(kv, "seed")
        clf = _kv_get(kv, "clf") or _kv_get(kv, "model")
        if subject is None or seed is None or clf is None:
            return None
        return make_exp_id(
            "official_faithful",
            subject=int(subject),
            seed=int(seed),
            clf=clf,
        )

    if prefix == "pilot":
        subject = _kv_get(kv, "subject") or _kv_get(kv, "subj")
        seed = _kv_get(kv, "seed")
        p_val = _kv_get(kv, "p")
        gen = _kv_get(kv, "gen") or _kv_get(kv, "gmodel")
        clf = _kv_get(kv, "clf")
        ratio = _kv_get(kv, "ratio") or _kv_get(kv, "r")
        if subject is None or seed is None or p_val is None or gen is None or clf is None or ratio is None:
            return None

        mode = None
        tag = None
        for item in extras:
            if item in {"baseline", "genaug"}:
                mode = item
            elif tag is None:
                tag = item

        meta = {
            "subject": int(subject),
            "seed": int(seed),
            "p": parse_p_tag(p_val),
            "gen": gen,
            "clf": clf,
            "ratio": parse_p_tag(ratio),
        }
        if tag:
            meta["tag"] = tag
        if mode is not None:
            return make_exp_id("pilot_clf", **meta, mode=mode)
        return make_exp_id("pilot", **meta)

    return None


def _collect_run_dirs(root: Path, subdir: str) -> list[Path]:
    base = root / "runs" / subdir
    if not base.exists():
        return []
    return [p for p in base.iterdir() if p.is_dir()]


def _apply_renames(renames: list[tuple[Path, Path]], dry_run: bool) -> None:
    for src, dst in renames:
        if src == dst:
            continue
        if dst.exists():
            print(f"[skip] target exists: {dst}")
            continue
        print(f"[rename] {src.name} -> {dst.name}")
        if not dry_run:
            src.rename(dst)


def main() -> None:
    ap = argparse.ArgumentParser(description="Migrate run directories to standardized naming.")
    ap.add_argument("--apply", action="store_true", help="Apply renames (default: dry-run)")
    ap.add_argument("--root", type=str, default="", help="Project root (default: auto)")
    args = ap.parse_args()

    root = Path(args.root) if args.root else project_root(__file__)

    track_map, aug_map, val_map = _build_paper_track_maps(root)
    pipeline_smoke_id = None
    try:
        data_cfg = load_yaml(root / "configs" / "data.yaml")
        split_cfg = load_yaml(root / "configs" / "split.yaml")
        subject = int(data_cfg.get("subjects", [1])[0])
        seed = int(split_cfg.get("seeds", [0])[0])
        p_val = float(split_cfg.get("low_data_fracs", [1.0])[0])
        pipeline_smoke_id = make_exp_id(
            "pipeline_validation_smoke",
            subject=subject,
            seed=seed,
            p=p_val,
        )
    except Exception:
        pipeline_smoke_id = None

    renames: list[tuple[Path, Path]] = []

    for path in _collect_run_dirs(root, "gen"):
        new_name = _rename_gen(path.name)
        if new_name and new_name != path.name:
            renames.append((path, path.parent / new_name))

        # Pilot generator runs use prefix "pilot" (handled below).
        elif path.name.startswith("pilot__"):
            new_name = _rename_clf(path.name, root, track_map, aug_map, val_map)
            if new_name and new_name != path.name:
                renames.append((path, path.parent / new_name))

    for path in _collect_run_dirs(root, "clf"):
        if path.name == "pipeline_validation_smoke" and pipeline_smoke_id:
            renames.append((path, path.parent / pipeline_smoke_id))
            continue
        new_name = _rename_clf(path.name, root, track_map, aug_map, val_map)
        if new_name and new_name != path.name:
            renames.append((path, path.parent / new_name))

    if not renames:
        print("No legacy run directories matched for renaming.")
        return

    dry_run = not bool(args.apply)
    if dry_run:
        print("[dry-run] showing planned renames (use --apply to execute)")

    _apply_renames(renames, dry_run=dry_run)


if __name__ == "__main__":
    from src.cli_deprecated import exit_deprecated
    exit_deprecated("migrate-runs")
