from __future__ import annotations

import torch
from torch import nn


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
        """Initialize EEGNet model.

        Inputs:
        - n_ch: number of EEG channels.
        - n_t: number of time samples per trial.
        - n_classes: number of output classes.
        - f1/d/f2/kernel_length/sep_kernel_length/dropout: EEGNet hyperparams.
        - max_norm_depthwise/max_norm_linear: max-norm constraints.

        Outputs:
        - EEGNet instance with conv feature extractor and linear classifier.

        Internal logic:
        - Builds temporal, spatial, and separable conv stacks then infers feature dim.
        """
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
        """Apply max-norm constraint to convolution weights.

        Inputs:
        - weight: conv weight tensor.
        - max_norm: maximum L2 norm.
        - dim: dimension for norm computation.

        Outputs:
        - None (in-place weight constraint).

        Internal logic:
        - Scales weights so their L2 norm along dim does not exceed max_norm.
        """
        if max_norm <= 0:
            return
        with torch.no_grad():
            w = weight.data
            norm = w.norm(2, dim=dim, keepdim=True)
            desired = torch.clamp(norm, max=max_norm)
            weight.data = w * (desired / (norm + 1e-8))

    @staticmethod
    def _apply_max_norm_linear(weight: torch.Tensor, max_norm: float) -> None:
        """Apply max-norm constraint to linear layer weights.

        Inputs:
        - weight: linear weight tensor.
        - max_norm: maximum L2 norm.

        Outputs:
        - None (in-place weight constraint).

        Internal logic:
        - Scales linear weights so column norms are clipped to max_norm.
        """
        if max_norm <= 0:
            return
        with torch.no_grad():
            w = weight.data
            norm = w.norm(2, dim=0, keepdim=True)
            desired = torch.clamp(norm, max=max_norm)
            weight.data = w * (desired / (norm + 1e-8))

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature extractor layers.

        Inputs:
        - x: torch.Tensor [B, 1, C, T]

        Outputs:
        - feature map tensor.

        Internal logic:
        - Applies temporal/spatial convs, pooling, and separable conv stack.
        """
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
        """Forward pass through EEGNet classifier.

        Inputs:
        - x: torch.Tensor [B, C, T]

        Outputs:
        - logits: torch.Tensor [B, K]

        Internal logic:
        - Applies feature extractor and a linear classifier with max-norm constraint.
        """
        self._apply_max_norm_linear(self.classifier.weight, self.max_norm_linear)
        z = self._forward_features(x.unsqueeze(1)).flatten(1)
        return self.classifier(z)


def build_eegnet(cfg: dict, n_ch: int, n_t: int, n_classes: int) -> EEGNet:
    """Construct EEGNet from a config dict.

    Inputs:
    - cfg: model config dict.
    - n_ch/n_t: input shape.
    - n_classes: output classes.

    Outputs:
    - EEGNet instance.

    Internal logic:
    - Maps config dict fields to EEGNet constructor arguments.
    """
    return EEGNet(
        n_ch=n_ch,
        n_t=n_t,
        n_classes=n_classes,
        f1=int(cfg.get("F1", 8)),
        d=int(cfg.get("D", 2)),
        f2=int(cfg.get("F2", 16)),
        kernel_length=int(cfg.get("kernel_length", 64)),
        sep_kernel_length=int(cfg.get("sep_kernel_length", 16)),
        dropout=float(cfg.get("dropout", 0.25)),
        max_norm_depthwise=float(cfg.get("max_norm_depthwise", 1.0)),
        max_norm_linear=float(cfg.get("max_norm_linear", 0.25)),
    )
