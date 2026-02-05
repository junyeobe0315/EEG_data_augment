from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score

from src.dataio import load_samples_by_ids
from src.models_clf import build_torch_classifier, normalize_classifier_type
from src.preprocess import ZScoreNormalizer
from src.utils import resolve_device


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def evaluate_saved_classifier(
    run_dir: str | Path,
    split: dict,
    index_df: pd.DataFrame,
    clf_cfg: dict,
    device: str = "auto",
) -> Dict[str, float]:
    run_dir = Path(run_dir)
    ckpt_pt = run_dir / "ckpt.pt"
    ckpt_pkl = run_dir / "ckpt.pkl"

    x_test, y_test = load_samples_by_ids(index_df, split["test_ids"])

    if ckpt_pkl.exists():
        obj = joblib.load(ckpt_pkl)
        norm = ZScoreNormalizer().load_state_dict(obj["normalizer"])
        x_test = norm.transform(x_test).astype(np.float32)

        svm = obj.get("svm_model")
        if svm is None:
            raise RuntimeError(f"SVM object missing in checkpoint: {ckpt_pkl}. Re-run with updated training code.")
        y_pred = svm.predict(x_test)
        return compute_metrics(y_test, y_pred)

    if ckpt_pt.exists():
        tdev = resolve_device(device)

        ckpt = torch.load(ckpt_pt, map_location=tdev, weights_only=False)
        model_type = normalize_classifier_type(str(ckpt.get("model_type", clf_cfg["model"].get("type", "eegnet"))))
        model = build_torch_classifier(
            model_type=model_type,
            n_ch=int(ckpt["shape"]["c"]),
            n_t=int(ckpt["shape"]["t"]),
            n_classes=int(ckpt["n_classes"]),
            cfg=clf_cfg["model"],
        ).to(tdev)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        norm = ZScoreNormalizer().load_state_dict(ckpt["normalizer"])
        x_test = norm.transform(x_test).astype(np.float32)

        xb = torch.from_numpy(x_test).to(tdev)
        with torch.no_grad():
            logits = model(xb)
            y_pred = logits.argmax(dim=1).cpu().numpy()
        return compute_metrics(y_test, y_pred)

    raise FileNotFoundError(f"No classifier checkpoint found in {run_dir}")
