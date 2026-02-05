from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist.

    Inputs:
    - path: directory path.

    Outputs:
    - Path object to the created/existing directory.

    Internal logic:
    - Converts to Path and calls mkdir with parents/exist_ok.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path) -> Any:
    """Read a JSON file into Python objects.

    Inputs:
    - path: JSON file path.

    Outputs:
    - Parsed object.

    Internal logic:
    - Opens file with UTF-8 and loads via json.load.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any, indent: int = 2) -> None:
    """Write a Python object to JSON.

    Inputs:
    - path: output file path.
    - obj: JSON-serializable object.
    - indent: indentation spaces.

    Outputs:
    - None (writes JSON file).

    Internal logic:
    - Ensures parent directory then dumps JSON with ASCII-safe encoding.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=True, indent=indent)


def read_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Read a .npz file into a dict of arrays.

    Inputs:
    - path: .npz file path.

    Outputs:
    - dict mapping array names to numpy arrays.

    Internal logic:
    - Uses numpy.load and returns a plain dict of arrays.
    """
    arr = np.load(path)
    return {k: arr[k] for k in arr.files}


def write_npz(path: str | Path, **arrays: np.ndarray) -> None:
    """Write arrays to a compressed .npz file.

    Inputs:
    - path: output path.
    - arrays: keyword arrays to save.

    Outputs:
    - None (writes .npz file).

    Internal logic:
    - Ensures parent directory then writes compressed npz.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **arrays)
