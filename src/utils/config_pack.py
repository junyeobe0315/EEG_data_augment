from __future__ import annotations

from pathlib import Path

from src.utils.config import load_yaml


def resolve_config_pack_path(base_path: str | Path, config_pack: str = "base") -> Path:
    """Resolve config path for a named config pack with fallback.

    Inputs:
    - base_path: canonical path under configs/.
    - config_pack: "base" or pack name (e.g., "tuned").

    Outputs:
    - Path to existing config file for the selected pack.

    Internal logic:
    - For non-base packs, checks configs/{pack}/... with the same suffix.
    - Falls back to base_path when pack file does not exist.
    """
    base = Path(base_path)
    pack = str(config_pack).strip().lower()
    if pack in {"", "base"}:
        return base

    try:
        rel = base.relative_to("configs")
    except ValueError:
        rel = base.name

    cand = Path("configs") / pack / rel
    return cand if cand.exists() else base


def load_yaml_with_pack(base_path: str | Path, config_pack: str = "base", overrides: list[str] | None = None) -> dict:
    """Load YAML from config pack with fallback to base."""
    resolved = resolve_config_pack_path(base_path, config_pack=config_pack)
    return load_yaml(resolved, overrides=overrides)
