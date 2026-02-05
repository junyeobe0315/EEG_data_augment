from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from src.models_clf import build_torch_classifier, normalize_classifier_type


def apply_normalizer_state(x: np.ndarray, state: dict) -> np.ndarray:
    mean = state["mean"]
    std = state["std"]
    return ((x - mean) / std).astype(np.float32)


def _extract_eegnet_features(model: torch.nn.Module, xb: torch.Tensor, model_type: str) -> torch.Tensor:
    mtype = normalize_classifier_type(model_type)
    if hasattr(model, "_forward_features"):
        if mtype == "eegnet":
            return model._forward_features(xb.unsqueeze(1)).flatten(1)
        if mtype == "eegnet_tf_faithful":
            return model._forward_features(xb).flatten(1)
    # Fallback: use logits as embedding.
    return model(xb)


class FrozenEEGNetEmbedder:
    def __init__(self, ckpt_path: str | Path, clf_cfg: dict, device: str = "cpu"):
        self.ckpt_path = Path(ckpt_path)
        self.device = str(device)
        self.ckpt = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        self.model_type = normalize_classifier_type(str(self.ckpt.get("model_type", "eegnet")))
        if self.model_type not in {"eegnet", "eegnet_tf_faithful"}:
            raise ValueError(f"Embedding checkpoint must be EEGNet family, got: {self.model_type}")

        cfg = dict(clf_cfg)
        cfg["type"] = self.model_type

        shape = self.ckpt["shape"]
        self.model = build_torch_classifier(
            model_type=self.model_type,
            n_ch=int(shape["c"]),
            n_t=int(shape["t"]),
            n_classes=int(self.ckpt["n_classes"]),
            cfg=cfg,
        ).to(self.device)
        self.model.load_state_dict(self.ckpt["state_dict"])  # type: ignore[arg-type]
        self.model.eval()
        self.norm_state = self.ckpt["normalizer"]

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return apply_normalizer_state(x, self.norm_state)

    @torch.no_grad()
    def transform(self, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
        feats = []
        for st in range(0, x.shape[0], batch_size):
            ed = min(st + batch_size, x.shape[0])
            xb = torch.from_numpy(x[st:ed]).to(self.device)
            z = _extract_eegnet_features(self.model, xb, model_type=self.model_type)
            feats.append(z.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(feats, axis=0)


def embed_with_frozen_eegnet(
    ckpt_path: str | Path,
    clf_cfg: dict,
    x: np.ndarray,
    device: str = "cpu",
    batch_size: int = 256,
) -> np.ndarray:
    embedder = FrozenEEGNetEmbedder(ckpt_path=ckpt_path, clf_cfg=clf_cfg, device=device)
    xn = embedder.normalize(x)
    return embedder.transform(xn, batch_size=batch_size)


def _median_heuristic_gamma(x: np.ndarray, y: np.ndarray) -> float:
    z = np.concatenate([x, y], axis=0)
    if len(z) > 1024:
        idx = np.random.choice(len(z), size=1024, replace=False)
        z = z[idx]
    d2 = np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=-1)
    med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
    sigma2 = max(float(med), 1e-6)
    return 1.0 / (2.0 * sigma2)


def _rbf_kernel(x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
    d2 = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    return np.exp(-gamma * d2)


def mmd_rbf(x: np.ndarray, y: np.ndarray, gamma: float | str = "median_heuristic") -> float:
    if len(x) < 2 or len(y) < 2:
        return math.nan

    if isinstance(gamma, str):
        gamma = _median_heuristic_gamma(x, y)

    kxx = _rbf_kernel(x, x, float(gamma))
    kyy = _rbf_kernel(y, y, float(gamma))
    kxy = _rbf_kernel(x, y, float(gamma))

    n = len(x)
    m = len(y)
    # Unbiased estimator.
    term_x = (np.sum(kxx) - np.trace(kxx)) / (n * (n - 1))
    term_y = (np.sum(kyy) - np.trace(kyy)) / (m * (m - 1))
    term_xy = (2.0 * np.sum(kxy)) / (n * m)
    return float(term_x + term_y - term_xy)


def sliced_wasserstein_distance(x: np.ndarray, y: np.ndarray, n_projections: int = 64, seed: int = 0) -> float:
    if len(x) == 0 or len(y) == 0:
        return math.nan

    rng = np.random.default_rng(seed)
    d = x.shape[1]
    dirs = rng.normal(size=(n_projections, d)).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8

    xp = x @ dirs.T
    yp = y @ dirs.T

    k = min(len(x), len(y))
    if k <= 1:
        return math.nan

    # Resample to the same size for 1D Wasserstein approximation.
    if len(x) != k:
        idx = rng.choice(len(x), size=k, replace=False)
        xp = xp[idx]
    if len(y) != k:
        idy = rng.choice(len(y), size=k, replace=False)
        yp = yp[idy]

    xp = np.sort(xp, axis=0)
    yp = np.sort(yp, axis=0)
    return float(np.mean(np.abs(xp - yp)))


def classwise_distance_summary(
    real_emb: np.ndarray,
    real_y: np.ndarray,
    aug_emb: np.ndarray,
    aug_y: np.ndarray,
    num_classes: int,
    n_projections: int = 64,
    mmd_gamma: float | str = "median_heuristic",
    seed: int = 0,
) -> dict:
    rows = {}
    swd_vals = []
    mmd_vals = []

    for cls in range(num_classes):
        rx = real_emb[real_y == cls]
        qx = aug_emb[aug_y == cls]

        swd = sliced_wasserstein_distance(rx, qx, n_projections=n_projections, seed=seed + cls)
        mmd = mmd_rbf(rx, qx, gamma=mmd_gamma)

        rows[f"dist_swd_class_{cls}"] = swd
        rows[f"dist_mmd_class_{cls}"] = mmd

        if not math.isnan(swd):
            swd_vals.append(swd)
        if not math.isnan(mmd):
            mmd_vals.append(mmd)

    rows["dist_swd"] = float(np.mean(swd_vals)) if swd_vals else math.nan
    rows["dist_mmd"] = float(np.mean(mmd_vals)) if mmd_vals else math.nan
    return rows
