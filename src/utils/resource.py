from __future__ import annotations

import subprocess
from typing import Tuple

import torch


def get_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def get_gpu_info() -> Tuple[str, str]:
    if not torch.cuda.is_available():
        return ("cpu", "0")
    name = torch.cuda.get_device_name(0)
    mem = str(torch.cuda.get_device_properties(0).total_memory)
    return (name, mem)
