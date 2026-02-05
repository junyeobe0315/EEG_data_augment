from __future__ import annotations

import numpy as np

from src.qc.qc_pipeline import filter_qc


def compute_target_counts(y_real: np.ndarray, alpha_ratio: float) -> dict[int, int]:
    """Compute per-class synthetic target counts given alpha_ratio = N_synth / N_real."""
    y_real = y_real.astype(np.int64)
    n_classes = int(y_real.max()) + 1
    counts = np.bincount(y_real, minlength=n_classes)
    targets = {}
    for cls in range(n_classes):
        targets[cls] = int(round(float(alpha_ratio) * int(counts[cls])))
    return targets


def build_synthetic_with_qc(
    sample_fn,
    target_counts: dict[int, int],
    qc_state: dict | None,
    qc_cfg: dict,
    sfreq: int,
    buffer: float = 1.2,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate synthetic data per class with optional QC filtering and oversampling."""
    xs = []
    ys = []
    report = {"class_counts_target": {}, "class_counts_actual": {}, "oversample_factor": 1.0}

    total_target = int(sum(target_counts.values()))
    max_attempt_factor = float(qc_cfg.get("max_attempt_factor", 3.0))
    max_attempts = max(1, int(np.ceil(max_attempt_factor * total_target)))

    total_attempted = 0
    total_kept = 0

    for cls, target in target_counts.items():
        target = int(target)
        report["class_counts_target"][str(cls)] = target
        if target <= 0:
            report["class_counts_actual"][str(cls)] = 0
            continue

        collected = []
        attempted = 0
        while len(collected) < target and total_attempted < max_attempts:
            remain = target - len(collected)
            n_try = max(remain, int(np.ceil(remain * buffer)))
            x_batch = sample_fn(int(cls), int(n_try))
            y_batch = np.full((x_batch.shape[0],), int(cls), dtype=np.int64)

            if qc_state is not None:
                mask, _ = filter_qc(x_batch, y_batch, sfreq=sfreq, cfg=qc_cfg, state=qc_state)
                x_keep = x_batch[mask]
            else:
                x_keep = x_batch

            collected.append(x_keep)
            attempted += int(x_batch.shape[0])
            total_attempted += int(x_batch.shape[0])

            if attempted >= max_attempts:
                break

        if collected:
            x_cls = np.concatenate(collected, axis=0)
        else:
            x_cls = np.empty((0,) + sample_fn(int(cls), 1).shape[1:], dtype=np.float32)

        if len(x_cls) > target:
            x_cls = x_cls[:target]

        y_cls = np.full((len(x_cls),), int(cls), dtype=np.int64)
        xs.append(x_cls)
        ys.append(y_cls)
        total_kept += int(len(x_cls))
        report["class_counts_actual"][str(cls)] = int(len(x_cls))

    report["n_before"] = int(total_attempted)
    report["n_after"] = int(total_kept)
    report["pass_rate"] = float(total_kept / max(1, total_attempted))
    report["oversample_factor"] = float(total_attempted / max(1, total_target))

    if xs:
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), report
    return np.empty((0,)), np.empty((0,)), report


def build_synthetic_pool(
    sample_fn,
    target_counts: dict[int, int],
    qc_state: dict | None,
    qc_cfg: dict,
    sfreq: int,
    buffer: float = 1.2,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build a reusable synthetic pool (optionally QC-filtered)."""
    x_pool, y_pool, report = build_synthetic_with_qc(
        sample_fn=sample_fn,
        target_counts=target_counts,
        qc_state=qc_state,
        qc_cfg=qc_cfg,
        sfreq=sfreq,
        buffer=buffer,
    )
    report["pool_size"] = int(len(x_pool))
    return x_pool, y_pool, report


def select_from_pool(
    x_pool: np.ndarray,
    y_pool: np.ndarray,
    target_counts: dict[int, int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Select class-wise subsets from a synthetic pool for a specific alpha_ratio."""
    rng = np.random.default_rng(int(seed))
    xs = []
    ys = []
    report = {
        "class_counts_pool": {},
        "class_counts_used": {},
        "pool_exhausted": False,
    }
    for cls, target in target_counts.items():
        cls = int(cls)
        target = int(target)
        idx = np.where(y_pool == cls)[0]
        report["class_counts_pool"][str(cls)] = int(len(idx))
        if target <= 0 or len(idx) == 0:
            report["class_counts_used"][str(cls)] = 0
            continue
        if len(idx) <= target:
            choose = idx
            report["pool_exhausted"] = True
        else:
            choose = rng.choice(idx, size=target, replace=False)
        x_sel = x_pool[choose]
        y_sel = np.full((len(x_sel),), cls, dtype=np.int64)
        xs.append(x_sel)
        ys.append(y_sel)
        report["class_counts_used"][str(cls)] = int(len(x_sel))

    if xs:
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), report
    return np.empty((0,)), np.empty((0,)), report
