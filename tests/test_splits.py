from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.splits import build_splits


def _make_index_df() -> pd.DataFrame:
    """Build a tiny synthetic index dataframe for split tests.

    Inputs:
    - None (uses fixed synthetic values).

    Outputs:
    - DataFrame with subject/session/label columns and fake file/row entries.

    Internal logic:
    - Enumerates sessions and labels to build a balanced fake dataset.
    """
    rows = []
    idx = 0
    for session in ["T", "E"]:
        for label in [0, 1, 2, 3]:
            for _ in range(10):
                rows.append(
                    {
                        "sample_id": f"S01_{session}_{idx:04d}",
                        "subject": 1,
                        "session": session,
                        "label": label,
                        "file": "dummy",
                        "row": idx,
                    }
                )
                idx += 1
    return pd.DataFrame(rows)


def test_build_splits_disjoint(tmp_path: Path) -> None:
    """Ensure train/val/test indices are disjoint.

    Inputs:
    - tmp_path: pytest temp directory for split outputs.

    Outputs:
    - Asserts no overlap across train/val/test index files.

    Internal logic:
    - Writes split files then loads them and checks set intersections.
    """
    index_df = _make_index_df()
    split_cfg = {"protocol": "cross_session", "val_ratio": 0.2, "seeds": [0], "low_data_fracs": [0.5]}

    build_splits(index_df, "bci2a", split_cfg, out_root=tmp_path)
    base = tmp_path / "bci2a" / "subject_01" / "seed0"
    tr = np.load(base / "T_train_full_idx.npy")
    va = np.load(base / "T_val_idx.npy")
    te = np.load(base / "E_test_idx.npy")

    assert len(set(tr) & set(va)) == 0
    assert len(set(tr) & set(te)) == 0
    assert len(set(va) & set(te)) == 0
