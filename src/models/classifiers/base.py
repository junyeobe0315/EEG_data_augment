from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from src.models.classifiers.eegnet import build_eegnet
from src.models.classifiers.eegconformer import build_eegconformer
from src.models.classifiers.ctnet import build_ctnet
from src.models.classifiers.fbcsp_svm import SVMClassifier


def normalize_classifier_type(model_type: str) -> str:
    key = model_type.strip().lower().replace("-", "_").replace(" ", "_")
    alias = {
        "eegnet": "eegnet",
        "eeg_conformer": "eegconformer",
        "conformer": "eegconformer",
        "eegconformer": "eegconformer",
        "ctnet": "ctnet",
        "svm": "svm",
        "fbcsp_svm": "svm",
    }
    if key not in alias:
        raise ValueError(f"Unsupported classifier type: {model_type}")
    return alias[key]


def is_sklearn_model(model_type: str) -> bool:
    return normalize_classifier_type(model_type) == "svm"


def build_classifier(
    model_type: str,
    n_ch: int,
    n_t: int,
    n_classes: int,
    cfg: dict,
) -> nn.Module | SVMClassifier:
    mtype = normalize_classifier_type(model_type)
    if mtype == "eegnet":
        return build_eegnet(cfg, n_ch, n_t, n_classes)
    if mtype == "eegconformer":
        return build_eegconformer(cfg, n_ch, n_t, n_classes)
    if mtype == "ctnet":
        return build_ctnet(cfg, n_ch, n_t, n_classes)
    if mtype == "svm":
        return SVMClassifier(
            c=float(cfg.get("C", 1.0)),
            kernel=str(cfg.get("kernel", "linear")),
            gamma=str(cfg.get("gamma", "scale")),
            sfreq=int(cfg.get("sfreq", 250)),
            bands=cfg.get("bands"),
            n_components=int(cfg.get("n_components", 4)),
        )
    raise ValueError(f"Unsupported classifier type: {model_type}")
