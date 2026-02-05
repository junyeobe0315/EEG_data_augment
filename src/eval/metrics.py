from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, balanced_accuracy_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute standard classification metrics.

    Inputs:
    - y_true: ndarray [N] ground-truth labels.
    - y_pred: ndarray [N] predicted labels.

    Outputs:
    - dict with acc, bal_acc, kappa, macro_f1.

    Internal logic:
    - Uses sklearn metrics and casts results to floats for JSON safety.
    """
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
