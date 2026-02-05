from __future__ import annotations

import numpy as np
import torch

from src.eval.metrics import compute_metrics
from src.models.classifiers.base import is_sklearn_model


def evaluate_classifier(model, x: np.ndarray, y: np.ndarray, device: str = "cpu") -> dict[str, float]:
    if is_sklearn_model(getattr(model, "model_type", "svm")):
        pred = model.predict(x)
        return compute_metrics(y, pred)

    model.eval()
    preds = []
    with torch.no_grad():
        for st in range(0, len(x), 256):
            xb = torch.from_numpy(x[st : st + 256]).to(device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds.append(pred)
    pred = np.concatenate(preds, axis=0)
    return compute_metrics(y, pred)
