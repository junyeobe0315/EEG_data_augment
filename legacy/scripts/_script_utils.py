from __future__ import annotations

import sys
from pathlib import Path


def project_root(file: str | Path) -> Path:
    root = Path(file).resolve().parents[1]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.append(root_str)
    return root
