from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Any

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Set seeds across Python, NumPy, and Torch for reproducibility.

    Inputs:
    - seed: integer seed.

    Outputs:
    - None (global RNG state is modified).

    Internal logic:
    - Sets Python, NumPy, and Torch seeds and forces deterministic CuDNN.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def stable_hash_seed(base_seed: int, payload: dict[str, Any], digest_len: int = 8) -> int:
    """Derive a stable seed from a base seed and payload.

    Inputs:
    - base_seed: base integer seed.
    - payload: dict to hash deterministically.
    - digest_len: hex length for hashing.

    Outputs:
    - deterministic integer seed.

    Internal logic:
    - Hashes the payload with SHA-256 and mixes it with base_seed.
    """
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    digest = int(hashlib.sha256(blob.encode("utf-8")).hexdigest()[:digest_len], 16)
    return int((int(base_seed) + digest) % (2**32 - 1))
