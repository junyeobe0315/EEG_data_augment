from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.data.normalize import ZScoreNormalizer
from src.models.classifiers.eegnet import EEGNet
from src.models.classifiers.base import normalize_classifier_type
from src.train.train_classifier import train_classifier
from src.utils.io import ensure_dir


def _extract_eegnet_features(model: EEGNet, xb: torch.Tensor) -> torch.Tensor:
    """Extract feature embeddings from EEGNet.

    Inputs:
    - model: EEGNet model.
    - xb: torch.Tensor [B, C, T].

    Outputs:
    - torch.Tensor [B, D] feature embeddings.

    Internal logic:
    - Uses EEGNet feature path when available, otherwise forward output.
    """
    # EEGNet expects [B, C, T]
    if hasattr(model, "_forward_features"):
        return model._forward_features(xb.unsqueeze(1)).flatten(1)
    return model(xb)


class FrozenEEGNetEmbedder:
    def __init__(self, ckpt_path: str | Path, device: str = "cpu"):
        """Load a frozen EEGNet embedding model from checkpoint.

        Inputs:
        - ckpt_path: path to a saved EEGNet classifier checkpoint.
        - device: torch device string.

        Outputs:
        - Embedder object with frozen model + normalizer.

        Internal logic:
        - Loads classifier checkpoint, rebuilds EEGNet, and freezes weights.
        """
        self.ckpt_path = Path(ckpt_path)
        self.device = str(device)
        try:
            ckpt = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(self.ckpt_path, map_location=self.device)
        self.ckpt = ckpt

        model_type = normalize_classifier_type(str(ckpt.get("model_type", "eegnet")))  # ensure EEGNet
        if model_type != "eegnet":
            raise ValueError(f"Embedding checkpoint must be EEGNet, got: {model_type}")

        shape = ckpt["shape"]
        cfg = ckpt.get("model_cfg", {})
        self.model = EEGNet(
            n_ch=int(shape["c"]),
            n_t=int(shape["t"]),
            n_classes=int(ckpt["n_classes"]),
            f1=int(cfg.get("F1", 8)),
            d=int(cfg.get("D", 2)),
            f2=int(cfg.get("F2", 16)),
            kernel_length=int(cfg.get("kernel_length", 64)),
            sep_kernel_length=int(cfg.get("sep_kernel_length", 16)),
            dropout=float(cfg.get("dropout", 0.25)),
            max_norm_depthwise=float(cfg.get("max_norm_depthwise", 1.0)),
            max_norm_linear=float(cfg.get("max_norm_linear", 0.25)),
        ).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])  # type: ignore[arg-type]
        self.model.eval()

        self.normalizer = ZScoreNormalizer.from_state(ckpt["normalizer"])

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Apply the stored normalizer to raw EEG.

        Inputs:
        - x: ndarray [N, C, T]

        Outputs:
        - normalized ndarray [N, C, T]

        Internal logic:
        - Delegates to the stored ZScoreNormalizer from the checkpoint.
        """
        return self.normalizer.transform(x)

    @torch.no_grad()
    def transform(self, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Transform EEG into embedding features.

        Inputs:
        - x: ndarray [N, C, T]
        - batch_size: inference batch size.

        Outputs:
        - ndarray [N, D] embeddings.

        Internal logic:
        - Runs forward passes in batches and concatenates feature vectors.
        """
        feats = []
        for st in range(0, x.shape[0], batch_size):
            ed = min(st + batch_size, x.shape[0])
            xb = torch.from_numpy(x[st:ed]).to(self.device)
            z = _extract_eegnet_features(self.model, xb)
            feats.append(z.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(feats, axis=0)


def train_embedding_eegnet(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    model_cfg: dict,
    train_cfg: dict,
    run_dir: str | Path,
    normalizer_state: dict,
    num_classes: int,
) -> Path:
    """Train EEGNet on real data and return checkpoint path for embedding.

    Inputs:
    - x_train/y_train: training data [N, C, T] / [N]
    - x_val/y_val: validation data [M, C, T] / [M]
    - model_cfg/train_cfg: EEGNet configs
    - run_dir: output directory
    - normalizer_state: fitted normalizer state
    - num_classes: number of classes

    Outputs:
    - Path to saved checkpoint (ckpt.pt).

    Internal logic:
    - Calls the classifier training loop with C0 settings and returns its checkpoint.
    """
    run_dir = ensure_dir(run_dir)
    _ = train_classifier(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_val,
        y_test=y_val,
        model_type="eegnet",
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        eval_cfg={"best_metric": "kappa", "best_direction": "max"},
        method="C0",
        alpha_ratio=0.0,
        num_classes=num_classes,
        run_dir=Path(run_dir),
        normalizer_state=normalizer_state,
        synth_data=None,
        evaluate_test=False,
    )
    return Path(run_dir) / "ckpt.pt"
