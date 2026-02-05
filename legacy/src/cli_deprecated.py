from __future__ import annotations

import sys


def exit_deprecated(command: str) -> None:
    msg = (
        f"[deprecated] Direct script execution is disabled. "
        f"Use: python main.py {command}"
    )
    print(msg, file=sys.stderr)
    raise SystemExit(2)
