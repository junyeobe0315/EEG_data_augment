from __future__ import annotations

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable


def run_subprocess_tasks(
    tasks: Iterable[dict],
    max_workers: int = 1,
    label: str | None = None,
) -> None:
    task_list = list(tasks)
    if not task_list:
        return

    def _run(task: dict) -> tuple[int, str]:
        cmd = task["cmd"]
        env = task.get("env")
        name = task.get("label", "task")
        merged_env = os.environ.copy()
        if env:
            merged_env.update({k: str(v) for k, v in env.items()})
        proc = subprocess.run(cmd, env=merged_env)
        return proc.returncode, name

    if max_workers <= 1:
        for t in task_list:
            code, name = _run(t)
            if code != 0:
                raise RuntimeError(f"{label or 'task'} failed: {name} (exit={code})")
        return

    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_run, t): t for t in task_list}
        for fut in as_completed(futures):
            code, name = fut.result()
            if code != 0:
                errors.append(f"{name} (exit={code})")

    if errors:
        raise RuntimeError(f"{label or 'task'} failures: {', '.join(errors)}")
