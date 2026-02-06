from __future__ import annotations

from pathlib import Path

from src.utils.config_pack import resolve_config_pack_path, load_yaml_with_pack


def test_config_pack_resolve_fallback(tmp_path: Path, monkeypatch) -> None:
    """tuned pack should fall back to base when tuned file is missing."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs" / "models").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs" / "models" / "eegnet.yaml").write_text("type: eegnet\nx: 1\n", encoding="utf-8")

    p = resolve_config_pack_path("configs/models/eegnet.yaml", config_pack="tuned")
    assert p.as_posix() == "configs/models/eegnet.yaml"


def test_config_pack_prefers_tuned_when_present(tmp_path: Path, monkeypatch) -> None:
    """tuned pack should use configs/tuned/... if file exists."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs" / "models").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs" / "tuned" / "models").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs" / "models" / "eegnet.yaml").write_text("type: eegnet\nx: 1\n", encoding="utf-8")
    (tmp_path / "configs" / "tuned" / "models" / "eegnet.yaml").write_text("type: eegnet\nx: 2\n", encoding="utf-8")

    cfg = load_yaml_with_pack("configs/models/eegnet.yaml", config_pack="tuned")
    assert cfg["x"] == 2
