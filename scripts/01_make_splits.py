#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from _script_utils import project_root

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = project_root(__file__)

from src.dataio import load_processed_index
from src.utils import ensure_dir, load_yaml, p_tag, save_json


def _stratified_subsample(ids: list[str], labels: list[int], frac: float, seed: int) -> list[str]:
    if frac >= 0.999:
        return list(ids)

    keep_ids = []
    rng = np.random.default_rng(seed)
    labels_arr = np.asarray(labels)
    ids_arr = np.asarray(ids)

    for cls in sorted(np.unique(labels_arr).tolist()):
        mask = labels_arr == cls
        cls_ids = ids_arr[mask]
        n = len(cls_ids)
        k = max(1, int(round(n * frac)))
        choose = rng.choice(n, size=min(k, n), replace=False)
        keep_ids.extend(cls_ids[choose].tolist())

    # keep stable order by original train id order
    keep_set = set(keep_ids)
    return [sid for sid in ids if sid in keep_set]


def main() -> None:
    data_cfg = load_yaml(ROOT / "configs/data.yaml")
    split_cfg = load_yaml(ROOT / "configs/split.yaml")
    sweep_cfg = load_yaml(ROOT / "configs/sweep.yaml")

    if split_cfg["protocol"] != "cross_session":
        raise ValueError("This script is now fixed for cross_session protocol.")
    if "test_ratio" in split_cfg:
        print("[info] cross_session protocol ignores split.test_ratio; E session is used as full test set.")

    split_dir = ensure_dir(ROOT / "data/splits")
    index_df = load_processed_index(data_cfg["index_path"])
    low_data_fracs = [float(p) for p in split_cfg.get("low_data_fracs", [1.0])]
    sweep_fracs = [float(p) for p in sweep_cfg.get("low_data_ratios", [])]
    if sweep_fracs and set(low_data_fracs) != set(sweep_fracs):
        print(
            "[warn] split.low_data_fracs and sweep.low_data_ratios differ. "
            "Using split.low_data_fracs for index generation."
        )

    n_saved = 0
    for subject in sorted(index_df["subject"].unique().tolist()):
        sub_df = index_df[index_df["subject"] == subject].copy()
        t_df = sub_df[sub_df["session"] == "T"].copy()
        e_df = sub_df[sub_df["session"] == "E"].copy()

        if len(t_df) == 0 or len(e_df) == 0:
            raise RuntimeError(f"Subject {subject}: both T and E sessions are required for cross_session.")

        for seed in split_cfg["seeds"]:
            tr_ids, va_ids = train_test_split(
                t_df["sample_id"].tolist(),
                test_size=float(split_cfg["val_ratio"]),
                random_state=int(seed),
                stratify=t_df["label"].tolist(),
            )
            te_ids = e_df["sample_id"].tolist()

            base = {
                "protocol": "cross_session",
                "subject": int(subject),
                "seed": int(seed),
                "train_ids": list(tr_ids),
                "val_ids": list(va_ids),
                "test_ids": list(te_ids),
            }
            base_path = split_dir / f"subject_{subject:02d}_seed_{int(seed)}.json"
            save_json(base_path, base)
            n_saved += 1

            # low-data variants: subsample only T_train
            tr_labels = t_df.set_index("sample_id").loc[tr_ids]["label"].tolist()
            for frac in low_data_fracs:
                sub_train_ids = _stratified_subsample(tr_ids, tr_labels, frac=frac, seed=int(seed))
                low = {
                    **base,
                    "low_data_frac": float(frac),
                    "train_ids": sub_train_ids,
                }
                tag = p_tag(frac)
                p_path = split_dir / f"subject_{subject:02d}_seed_{int(seed)}_p_{tag}.json"
                save_json(p_path, low)
                n_saved += 1

    print(f"Saved {n_saved} split files under {split_dir}")


if __name__ == "__main__":
    main()
