from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from mne.decoding import CSP
from scipy.signal import butter, sosfiltfilt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn


def _norm_model_type(model_type: str) -> str:
    key = model_type.strip().lower().replace("-", "_").replace(" ", "_")
    alias = {
        "eegnet": "eegnet",
        "eegnet_tf_faithful": "eegnet_tf_faithful",
        "eegnet_official": "eegnet_tf_faithful",
        "svm": "svm",
        "eeg_conformer": "eeg_conformer",
        "conformer": "eeg_conformer",
        "atcnet": "atcnet",
        "atcnet_tf_faithful": "atcnet_tf_faithful",
        "atcnet_official": "atcnet_tf_faithful",
        "ctnet": "ctnet",
        "ct_net": "ctnet",
    }
    if key not in alias:
        raise ValueError(f"Unsupported classifier model type: {model_type}")
    return alias[key]


class EEGNet(nn.Module):
    """EEGNet-v2 style classifier (Lawhern et al., 2018)."""

    def __init__(
        self,
        n_ch: int,
        n_t: int,
        n_classes: int,
        f1: int = 8,
        d: int = 2,
        f2: int = 16,
        kernel_length: int = 64,
        sep_kernel_length: int = 16,
        dropout: float = 0.25,
        max_norm_depthwise: float = 1.0,
        max_norm_linear: float = 0.25,
    ):
        super().__init__()
        self.max_norm_depthwise = float(max_norm_depthwise)
        self.max_norm_linear = float(max_norm_linear)
        self.temporal = nn.Conv2d(1, f1, kernel_size=(1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(f1)

        self.spatial = nn.Conv2d(f1, f1 * d, kernel_size=(n_ch, 1), groups=f1, bias=False)
        self.bn2 = nn.BatchNorm2d(f1 * d)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        self.sep_dw = nn.Conv2d(
            f1 * d,
            f1 * d,
            kernel_size=(1, sep_kernel_length),
            padding=(0, sep_kernel_length // 2),
            groups=f1 * d,
            bias=False,
        )
        self.sep_pw = nn.Conv2d(f1 * d, f2, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(f2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)

        self.act = nn.ELU()

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_ch, n_t)
            feat_dim = self._forward_features(dummy).flatten(1).shape[1]
        self.classifier = nn.Linear(feat_dim, n_classes)

    @staticmethod
    def _apply_max_norm_conv2d(weight: torch.Tensor, max_norm: float, dim: int = 0) -> None:
        if max_norm <= 0:
            return
        with torch.no_grad():
            w = weight.data
            norm = w.norm(2, dim=dim, keepdim=True)
            desired = torch.clamp(norm, max=max_norm)
            weight.data = w * (desired / (norm + 1e-8))

    @staticmethod
    def _apply_max_norm_linear(weight: torch.Tensor, max_norm: float) -> None:
        if max_norm <= 0:
            return
        with torch.no_grad():
            w = weight.data
            norm = w.norm(2, dim=0, keepdim=True)
            desired = torch.clamp(norm, max=max_norm)
            weight.data = w * (desired / (norm + 1e-8))

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        self._apply_max_norm_conv2d(self.spatial.weight, self.max_norm_depthwise, dim=1)
        z = self.temporal(x)
        z = self.bn1(z)

        z = self.spatial(z)
        z = self.bn2(z)
        z = self.act(z)
        z = self.pool1(z)
        z = self.drop1(z)

        z = self.sep_dw(z)
        z = self.sep_pw(z)
        z = self.bn3(z)
        z = self.act(z)
        z = self.pool2(z)
        z = self.drop2(z)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        self._apply_max_norm_linear(self.classifier.weight, self.max_norm_linear)
        z = self._forward_features(x.unsqueeze(1)).flatten(1)
        return self.classifier(z)


class _TransformerBlock(nn.Module):
    def __init__(self, emb_size: int, n_heads: int, dropout: float, ff_mult: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = nn.MultiheadAttention(emb_size, n_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(emb_size)
        self.ff = nn.Sequential(
            nn.Linear(emb_size, emb_size * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_size * ff_mult, emb_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm1(x)
        attn_out, _ = self.attn(z, z, z, need_weights=False)
        x = x + self.drop1(attn_out)
        x = x + self.ff(self.norm2(x))
        return x


class _TransformerEncoder(nn.Module):
    def __init__(self, emb_size: int, n_heads: int, depth: int, dropout: float, ff_mult: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList([_TransformerBlock(emb_size, n_heads, dropout, ff_mult=ff_mult) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


class EEGConformer(nn.Module):
    """EEG-Conformer-like classifier using shallow conv patch embedding + Transformer."""

    def __init__(
        self,
        n_ch: int,
        n_t: int,
        n_classes: int,
        conv_channels: int = 40,
        emb_size: int = 40,
        n_heads: int = 10,
        n_layers: int = 6,
        dropout: float = 0.5,
        temporal_kernel: int = 25,
        pool_kernel: int = 75,
        pool_stride: int = 15,
        fc1: int = 256,
        fc2: int = 32,
    ):
        super().__init__()
        self.patch = nn.Sequential(
            nn.Conv2d(1, conv_channels, kernel_size=(1, temporal_kernel), stride=(1, 1), bias=False),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=(n_ch, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(conv_channels),
            nn.ELU(),
            nn.AvgPool2d((1, pool_kernel), stride=(1, pool_stride)),
            nn.Dropout(dropout),
            nn.Conv2d(conv_channels, emb_size, kernel_size=(1, 1), stride=(1, 1), bias=False),
        )
        self.encoder = _TransformerEncoder(emb_size=emb_size, n_heads=n_heads, depth=n_layers, dropout=dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_ch, n_t)
            feat = self.patch(dummy).flatten(1)
            flat_dim = int(feat.shape[1])

        self.head = nn.Sequential(
            nn.Linear(flat_dim, fc1),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(fc1, fc2),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(fc2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        z = self.patch(x.unsqueeze(1)).squeeze(2)  # [B, E, L]
        z = z.transpose(1, 2)  # [B, L, E]
        z = self.encoder(z)
        return self.head(z.flatten(1))


class _CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation, padding=self.pad, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x)
        if self.pad > 0:
            z = z[..., :-self.pad]
        return z


class _ATCTCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = _CausalConv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.act1 = nn.ELU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = _CausalConv1d(out_ch, out_ch, kernel_size=kernel_size, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act2 = nn.ELU()
        self.drop2 = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.drop1(self.act1(self.bn1(self.conv1(x))))
        z = self.drop2(self.act2(self.bn2(self.conv2(z))))
        return self.act2(z + self.skip(x))


class ATCNet(nn.Module):
    """ATCNet-style architecture close to the public reference implementation."""

    def __init__(
        self,
        n_ch: int,
        n_t: int,
        n_classes: int,
        f1: int = 16,
        d: int = 2,
        f2: int = 32,
        kernel_length: int = 64,
        sep_kernel_length: int = 16,
        pool1: int = 8,
        pool2: int = 7,
        num_windows: int = 5,
        tcn_depth: int = 2,
        tcn_kernel_size: int = 4,
        tcn_filters: int = 32,
        n_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * d, kernel_size=(n_ch, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.AvgPool2d((1, pool1)),
            nn.Dropout(dropout),
            nn.Conv2d(f1 * d, f2, kernel_size=(1, sep_kernel_length), padding="same", bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pool2)),
            nn.Dropout(dropout),
        )
        self.num_windows = int(num_windows)
        self.attn = nn.MultiheadAttention(embed_dim=f2, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(f2)

        tcn = []
        in_ch = f2
        for i in range(int(tcn_depth)):
            tcn.append(_ATCTCNBlock(in_ch=in_ch, out_ch=tcn_filters, kernel_size=tcn_kernel_size, dilation=2**i, dropout=dropout))
            in_ch = tcn_filters
        self.tcn = nn.Sequential(*tcn)
        self.cls = nn.Linear(tcn_filters, n_classes)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_ch, n_t)
            _ = self.frontend(dummy)

    def _iter_windows(self, z: torch.Tensor) -> list[torch.Tensor]:
        # z: [B, L, F]
        if self.num_windows <= 1 or z.shape[1] <= self.num_windows:
            return [z]
        outs = []
        l = z.shape[1]
        for i in range(self.num_windows):
            st = i
            ed = l - self.num_windows + i + 1
            if ed <= st:
                outs.append(z)
            else:
                outs.append(z[:, st:ed, :])
        return outs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        z = self.frontend(x.unsqueeze(1)).squeeze(2).transpose(1, 2)  # [B, L, F2]
        logits = []
        for w in self._iter_windows(z):
            a, _ = self.attn(w, w, w, need_weights=False)
            a = self.attn_norm(w + a)
            t = self.tcn(a.transpose(1, 2))
            logits.append(self.cls(t[..., -1]))
        return torch.stack(logits, dim=0).mean(dim=0)


class CTNet(nn.Module):
    """CTNet-like classifier (EEGNet-style tokenization + positional Transformer)."""

    def __init__(
        self,
        n_ch: int,
        n_t: int,
        n_classes: int,
        emb_size: int = 40,
        n_heads: int = 8,
        n_layers: int = 3,
        f1: int = 20,
        d: int = 2,
        kernel_length: int = 64,
        sep_kernel_length: int = 16,
        pool1: int = 8,
        pool2: int = 8,
        dropout: float = 0.3,
        pos_dropout: float = 0.1,
        ff_mult: int = 4,
    ):
        super().__init__()
        f2 = f1 * d
        self.emb_size = int(emb_size)
        self.patch = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f2, kernel_size=(n_ch, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pool1)),
            nn.Dropout(dropout),
            nn.Conv2d(f2, f2, kernel_size=(1, sep_kernel_length), padding="same", bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pool2)),
            nn.Dropout(dropout),
            nn.Conv2d(f2, emb_size, kernel_size=(1, 1), bias=False),
        )
        self.pos_drop = nn.Dropout(pos_dropout)
        self.encoder = _TransformerEncoder(emb_size=emb_size, n_heads=n_heads, depth=n_layers, dropout=dropout, ff_mult=ff_mult)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_ch, n_t)
            tokens = self.patch(dummy).squeeze(2).transpose(1, 2)  # [1, L, E]
            n_tokens = int(tokens.shape[1])
            flat_dim = int(tokens.flatten(1).shape[1])

        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, emb_size))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(flat_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        z = self.patch(x.unsqueeze(1)).squeeze(2).transpose(1, 2)  # [B, L, E]
        z = z * math.sqrt(self.emb_size)
        z = self.pos_drop(z + self.pos_embed[:, : z.shape[1], :])
        z = self.encoder(z)
        return self.classifier(z.flatten(1))


class FBCSPExtractor:
    """Filter-Bank CSP feature extractor for MI decoding."""

    def __init__(self, sfreq: int = 250, bands: list[list[float]] | None = None, n_components: int = 4):
        self.sfreq = int(sfreq)
        self.bands = bands or [[4, 8], [8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, 32], [32, 36], [36, 40]]
        self.n_components = int(n_components)
        self.csp_list: list[CSP] = []

    def _bandpass(self, x: np.ndarray, l: float, h: float) -> np.ndarray:
        sos = butter(4, [l, h], btype="bandpass", fs=self.sfreq, output="sos")
        return sosfiltfilt(sos, x, axis=-1).astype(np.float32)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "FBCSPExtractor":
        self.csp_list = []
        for l, h in self.bands:
            xb = self._bandpass(x, float(l), float(h))
            csp = CSP(n_components=self.n_components, reg=None, log=True, norm_trace=False, cov_est="epoch")
            csp.fit(xb, y)
            self.csp_list.append(csp)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self.csp_list:
            raise RuntimeError("FBCSPExtractor must be fitted before transform")
        feats = []
        for (l, h), csp in zip(self.bands, self.csp_list):
            xb = self._bandpass(x, float(l), float(h))
            feats.append(csp.transform(xb))
        return np.concatenate(feats, axis=1).astype(np.float32)


@dataclass
class SVMClassifier:
    c: float = 1.0
    kernel: str = "linear"
    gamma: str = "scale"
    sfreq: int = 250
    bands: list[list[float]] = field(default_factory=lambda: [[4, 8], [8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, 32], [32, 36], [36, 40]])
    n_components: int = 4

    def __post_init__(self) -> None:
        self.extractor = FBCSPExtractor(sfreq=self.sfreq, bands=self.bands, n_components=self.n_components)
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svc", SVC(C=self.c, kernel=self.kernel, gamma=self.gamma, class_weight="balanced")),
            ]
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        feat = self.extractor.fit(x, y).transform(x)
        self.pipeline.fit(feat, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        feat = self.extractor.transform(x)
        return self.pipeline.predict(feat)


def build_torch_classifier(
    model_type: str,
    n_ch: int,
    n_t: int,
    n_classes: int,
    cfg: dict,
) -> nn.Module:
    from src.models_official_faithful import build_faithful_model

    mtype = _norm_model_type(model_type)

    if mtype == "eegnet_tf_faithful":
        mc = cfg.get("eegnet_tf_faithful", cfg.get("eegnet", {}))
        return build_faithful_model(
            model_key="eegnet_tf_faithful",
            n_ch=n_ch,
            n_t=n_t,
            n_classes=n_classes,
            model_cfg=mc,
        )

    if mtype == "eegnet":
        mc = cfg.get("eegnet", {})
        return EEGNet(
            n_ch=n_ch,
            n_t=n_t,
            n_classes=n_classes,
            f1=int(mc.get("F1", 8)),
            d=int(mc.get("D", 2)),
            f2=int(mc.get("F2", 16)),
            kernel_length=int(mc.get("kernel_length", 64)),
            sep_kernel_length=int(mc.get("sep_kernel_length", 16)),
            dropout=float(mc.get("dropout", 0.25)),
            max_norm_depthwise=float(mc.get("max_norm_depthwise", 1.0)),
            max_norm_linear=float(mc.get("max_norm_linear", 0.25)),
        )

    if mtype == "eeg_conformer":
        mc = cfg.get("eeg_conformer", {})
        return EEGConformer(
            n_ch=n_ch,
            n_t=n_t,
            n_classes=n_classes,
            conv_channels=int(mc.get("conv_channels", 40)),
            emb_size=int(mc.get("emb_size", 40)),
            n_heads=int(mc.get("n_heads", 10)),
            n_layers=int(mc.get("n_layers", 6)),
            dropout=float(mc.get("dropout", 0.5)),
            temporal_kernel=int(mc.get("temporal_kernel", 25)),
            pool_kernel=int(mc.get("pool_kernel", 75)),
            pool_stride=int(mc.get("pool_stride", 15)),
            fc1=int(mc.get("fc1", 256)),
            fc2=int(mc.get("fc2", 32)),
        )

    if mtype == "atcnet":
        mc = cfg.get("atcnet", {})
        return ATCNet(
            n_ch=n_ch,
            n_t=n_t,
            n_classes=n_classes,
            f1=int(mc.get("F1", 16)),
            d=int(mc.get("D", 2)),
            f2=int(mc.get("F2", 32)),
            kernel_length=int(mc.get("kernel_length", 64)),
            sep_kernel_length=int(mc.get("sep_kernel_length", 16)),
            pool1=int(mc.get("pool1", 8)),
            pool2=int(mc.get("pool2", 7)),
            num_windows=int(mc.get("num_windows", 5)),
            tcn_depth=int(mc.get("tcn_depth", 2)),
            tcn_kernel_size=int(mc.get("tcn_kernel_size", 4)),
            tcn_filters=int(mc.get("tcn_filters", 32)),
            n_heads=int(mc.get("n_heads", 4)),
            dropout=float(mc.get("dropout", 0.3)),
        )

    if mtype == "atcnet_tf_faithful":
        mc = cfg.get("atcnet_tf_faithful", cfg.get("atcnet", {}))
        return build_faithful_model(
            model_key="atcnet_tf_faithful",
            n_ch=n_ch,
            n_t=n_t,
            n_classes=n_classes,
            model_cfg=mc,
        )

    if mtype == "ctnet":
        mc = cfg.get("ctnet", {})
        return CTNet(
            n_ch=n_ch,
            n_t=n_t,
            n_classes=n_classes,
            emb_size=int(mc.get("emb_size", 40)),
            n_heads=int(mc.get("n_heads", 8)),
            n_layers=int(mc.get("n_layers", 3)),
            f1=int(mc.get("F1", 20)),
            d=int(mc.get("D", 2)),
            kernel_length=int(mc.get("kernel_length", 64)),
            sep_kernel_length=int(mc.get("sep_kernel_length", 16)),
            pool1=int(mc.get("pool1", 8)),
            pool2=int(mc.get("pool2", 8)),
            dropout=float(mc.get("dropout", 0.3)),
            pos_dropout=float(mc.get("pos_dropout", 0.1)),
            ff_mult=int(mc.get("ff_mult", 4)),
        )

    raise ValueError(f"build_torch_classifier only supports torch models, got: {model_type}")


def build_svm_classifier(cfg: dict) -> SVMClassifier:
    sv = cfg.get("svm", {})
    return SVMClassifier(
        c=float(sv.get("C", 1.0)),
        kernel=str(sv.get("kernel", "linear")),
        gamma=str(sv.get("gamma", "scale")),
        sfreq=int(sv.get("sfreq", 250)),
        bands=sv.get("bands", [[4, 8], [8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, 32], [32, 36], [36, 40]]),
        n_components=int(sv.get("n_components", 4)),
    )


def is_sklearn_model(model_type: str) -> bool:
    return _norm_model_type(model_type) == "svm"


def normalize_classifier_type(model_type: str) -> str:
    return _norm_model_type(model_type)
