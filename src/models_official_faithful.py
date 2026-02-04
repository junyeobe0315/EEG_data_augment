from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _max_norm_(weight: torch.Tensor, max_norm: float, dims: tuple[int, ...]) -> None:
    if max_norm <= 0:
        return
    with torch.no_grad():
        w = weight.data
        norm = torch.linalg.vector_norm(w, ord=2, dim=dims, keepdim=True)
        desired = torch.clamp(norm, max=max_norm)
        weight.data = w * (desired / (norm + 1e-8))


class _CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.pad,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x)
        if self.pad > 0:
            z = z[..., :-self.pad]
        return z


class EEGNetOfficialFaithful(nn.Module):
    """
    PyTorch counterpart of EEG-ATCNet repository `EEGNet_classifier`.
    Input: [B, C, T], output: logits [B, n_classes].
    """

    def __init__(
        self,
        n_ch: int,
        n_t: int,
        n_classes: int,
        f1: int = 8,
        d: int = 2,
        kernel_length: int = 64,
        dropout: float = 0.25,
        max_norm_depthwise: float = 1.0,
        max_norm_linear: float = 0.25,
    ):
        super().__init__()
        self.max_norm_depthwise = float(max_norm_depthwise)
        self.max_norm_linear = float(max_norm_linear)

        f2 = int(f1 * d)
        self.conv1 = nn.Conv2d(1, f1, kernel_size=(kernel_length, 1), padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(f1)

        # Keras DepthwiseConv2D((1, Chans), depth_multiplier=D)
        self.depthwise = nn.Conv2d(f1, f2, kernel_size=(1, n_ch), groups=f1, bias=False)
        self.bn2 = nn.BatchNorm2d(f2)
        self.pool1 = nn.AvgPool2d((8, 1))
        self.drop1 = nn.Dropout(dropout)

        # Keras SeparableConv2D(F2, (16, 1))
        self.sep_dw = nn.Conv2d(f2, f2, kernel_size=(16, 1), padding="same", groups=f2, bias=False)
        self.sep_pw = nn.Conv2d(f2, f2, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(f2)
        self.pool2 = nn.AvgPool2d((8, 1))
        self.drop2 = nn.Dropout(dropout)
        self.act = nn.ELU()

        with torch.no_grad():
            dummy = torch.zeros(1, n_ch, n_t)
            feat_dim = self._forward_features(dummy).flatten(1).shape[1]
        self.classifier = nn.Linear(feat_dim, n_classes, bias=True)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # Convert [B, C, T] -> [B, 1, T, C] to match Keras channels_last after Permute.
        z = x.unsqueeze(1).permute(0, 1, 3, 2)
        z = self.conv1(z)
        z = self.bn1(z)

        _max_norm_(self.depthwise.weight, self.max_norm_depthwise, dims=(1, 2, 3))
        z = self.depthwise(z)
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
        _max_norm_(self.classifier.weight, self.max_norm_linear, dims=(1,))
        z = self._forward_features(x).flatten(1)
        return self.classifier(z)


class _ATCNetConvBlock(nn.Module):
    def __init__(
        self,
        n_ch: int,
        f1: int,
        d: int,
        kernel_length: int,
        pool_size: int,
        dropout: float,
        max_norm_conv: float,
    ):
        super().__init__()
        f2 = int(f1 * d)
        self.max_norm_conv = float(max_norm_conv)

        self.conv1 = nn.Conv2d(1, f1, kernel_size=(kernel_length, 1), padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(f1)

        self.conv2 = nn.Conv2d(f1, f2, kernel_size=(1, n_ch), groups=f1, bias=False)
        self.bn2 = nn.BatchNorm2d(f2)
        self.pool1 = nn.AvgPool2d((8, 1))
        self.drop1 = nn.Dropout(dropout)

        self.conv3 = nn.Conv2d(f2, f2, kernel_size=(16, 1), padding="same", bias=False)
        self.bn3 = nn.BatchNorm2d(f2)
        self.pool2 = nn.AvgPool2d((pool_size, 1))
        self.drop2 = nn.Dropout(dropout)
        self.act = nn.ELU()

    def regularized_layers(self) -> list[nn.Module]:
        return [self.conv1, self.conv2, self.conv3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _max_norm_(self.conv1.weight, self.max_norm_conv, dims=(1, 2, 3))
        z = self.conv1(x)
        z = self.bn1(z)

        _max_norm_(self.conv2.weight, self.max_norm_conv, dims=(1, 2, 3))
        z = self.conv2(z)
        z = self.bn2(z)
        z = self.act(z)
        z = self.pool1(z)
        z = self.drop1(z)

        _max_norm_(self.conv3.weight, self.max_norm_conv, dims=(1, 2, 3))
        z = self.conv3(z)
        z = self.bn3(z)
        z = self.act(z)
        z = self.pool2(z)
        z = self.drop2(z)
        return z


class _ATCNetTCNBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        depth: int,
        kernel_size: int,
        filters: int,
        dropout: float,
        max_norm_conv: float,
    ):
        super().__init__()
        self.depth = int(depth)
        self.max_norm_conv = float(max_norm_conv)

        self.conv_a1 = _CausalConv1d(input_dim, filters, kernel_size=kernel_size, dilation=1, bias=False)
        self.bn_a1 = nn.BatchNorm1d(filters)
        self.conv_a2 = _CausalConv1d(filters, filters, kernel_size=kernel_size, dilation=1, bias=False)
        self.bn_a2 = nn.BatchNorm1d(filters)

        self.skip = nn.Conv1d(input_dim, filters, kernel_size=1, bias=False) if input_dim != filters else nn.Identity()

        self.conv_b1 = nn.ModuleList()
        self.bn_b1 = nn.ModuleList()
        self.conv_b2 = nn.ModuleList()
        self.bn_b2 = nn.ModuleList()
        for i in range(self.depth - 1):
            dil = 2 ** (i + 1)
            self.conv_b1.append(_CausalConv1d(filters, filters, kernel_size=kernel_size, dilation=dil, bias=False))
            self.bn_b1.append(nn.BatchNorm1d(filters))
            self.conv_b2.append(_CausalConv1d(filters, filters, kernel_size=kernel_size, dilation=dil, bias=False))
            self.bn_b2.append(nn.BatchNorm1d(filters))

        self.drop = nn.Dropout(dropout)
        self.act = nn.ELU()

    def regularized_layers(self) -> list[nn.Module]:
        layers: list[nn.Module] = [self.conv_a1.conv, self.conv_a2.conv]
        if isinstance(self.skip, nn.Conv1d):
            layers.append(self.skip)
        for c1, c2 in zip(self.conv_b1, self.conv_b2):
            layers.extend([c1.conv, c2.conv])
        return layers

    def _apply_tcn_constraints(self) -> None:
        _max_norm_(self.conv_a1.conv.weight, self.max_norm_conv, dims=(1, 2))
        _max_norm_(self.conv_a2.conv.weight, self.max_norm_conv, dims=(1, 2))
        if isinstance(self.skip, nn.Conv1d):
            _max_norm_(self.skip.weight, self.max_norm_conv, dims=(1, 2))
        for c1, c2 in zip(self.conv_b1, self.conv_b2):
            _max_norm_(c1.conv.weight, self.max_norm_conv, dims=(1, 2))
            _max_norm_(c2.conv.weight, self.max_norm_conv, dims=(1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, F]
        self._apply_tcn_constraints()
        z = x.transpose(1, 2)  # [B, F, L]

        a = self.drop(self.act(self.bn_a1(self.conv_a1(z))))
        a = self.drop(self.act(self.bn_a2(self.conv_a2(a))))
        skip = self.skip(z)
        out = self.act(a + skip)

        for conv1, bn1, conv2, bn2 in zip(self.conv_b1, self.bn_b1, self.conv_b2, self.bn_b2):
            b = self.drop(self.act(bn1(conv1(out))))
            b = self.drop(self.act(bn2(conv2(b))))
            out = self.act(b + out)

        return out.transpose(1, 2)  # [B, L, F]


class ATCNetOfficialFaithful(nn.Module):
    """
    PyTorch counterpart of EEG-ATCNet `ATCNet_` with `attention='mha', fuse='average'`.
    Input: [B, C, T], output: logits [B, n_classes].
    """

    def __init__(
        self,
        n_ch: int,
        n_t: int,
        n_classes: int,
        n_windows: int = 5,
        eegn_f1: int = 16,
        eegn_d: int = 2,
        eegn_kernel_size: int = 64,
        eegn_pool_size: int = 7,
        eegn_dropout: float = 0.3,
        tcn_depth: int = 2,
        tcn_kernel_size: int = 4,
        tcn_filters: int = 32,
        tcn_dropout: float = 0.3,
        conv_weight_decay: float = 0.009,
        dense_weight_decay: float = 0.5,
        conv_max_norm: float = 0.6,
    ):
        super().__init__()
        self.n_windows = int(n_windows)
        self.conv_weight_decay = float(conv_weight_decay)
        self.dense_weight_decay = float(dense_weight_decay)

        self.conv_block = _ATCNetConvBlock(
            n_ch=n_ch,
            f1=eegn_f1,
            d=eegn_d,
            kernel_length=eegn_kernel_size,
            pool_size=eegn_pool_size,
            dropout=eegn_dropout,
            max_norm_conv=conv_max_norm,
        )
        f2 = int(eegn_f1 * eegn_d)

        # Keras attention block: LayerNorm -> MHA(key_dim=8, num_heads=2, dropout=0.5) -> Dropout(0.3) -> residual add.
        self.attn_norm = nn.LayerNorm(f2, eps=1e-6)
        self.attn = nn.MultiheadAttention(embed_dim=f2, num_heads=2, dropout=0.5, batch_first=True)
        self.attn_drop = nn.Dropout(0.3)

        self.tcn = _ATCNetTCNBlock(
            input_dim=f2,
            depth=tcn_depth,
            kernel_size=tcn_kernel_size,
            filters=tcn_filters,
            dropout=tcn_dropout,
            max_norm_conv=conv_max_norm,
        )
        self.heads = nn.ModuleList([nn.Linear(tcn_filters, n_classes, bias=True) for _ in range(self.n_windows)])

        with torch.no_grad():
            dummy = torch.zeros(1, n_ch, n_t)
            _ = self.forward(dummy)

    def _windows(self, x: torch.Tensor) -> list[torch.Tensor]:
        # x: [B, L, F]
        outs: list[torch.Tensor] = []
        l = int(x.shape[1])
        for i in range(self.n_windows):
            st = int(i)
            ed = int(l - self.n_windows + i + 1)
            if ed <= st:
                outs.append(x)
            else:
                outs.append(x[:, st:ed, :])
        return outs

    def regularization_loss(self) -> torch.Tensor:
        reg = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.conv_block.regularized_layers():
            reg = reg + layer.weight.pow(2).sum() * self.conv_weight_decay
        for layer in self.tcn.regularized_layers():
            reg = reg + layer.weight.pow(2).sum() * self.conv_weight_decay
        for head in self.heads:
            reg = reg + head.weight.pow(2).sum() * self.dense_weight_decay
        return reg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, T] -> [B, 1, T, C]
        z = x.unsqueeze(1).permute(0, 1, 3, 2)
        z = self.conv_block(z)
        # Keras Lambda(lambda x: x[:, :, -1, :]) after conv block tensor [B, T', 1, F2]
        z = z.squeeze(3).transpose(1, 2)  # [B, T', F2]

        win_logits = []
        for i, w in enumerate(self._windows(z)):
            q = self.attn_norm(w)
            a, _ = self.attn(q, q, q, need_weights=False)
            w_att = w + self.attn_drop(a)
            t = self.tcn(w_att)
            last = t[:, -1, :]
            win_logits.append(self.heads[i](last))

        return torch.stack(win_logits, dim=0).mean(dim=0)


@dataclass
class FaithfulModelSpec:
    key: str
    target_acc: float


def build_faithful_model(model_key: str, n_ch: int, n_t: int, n_classes: int, model_cfg: dict) -> nn.Module:
    key = str(model_key).strip().lower()
    if key == "eegnet_tf_faithful":
        return EEGNetOfficialFaithful(
            n_ch=n_ch,
            n_t=n_t,
            n_classes=n_classes,
            f1=int(model_cfg.get("f1", 8)),
            d=int(model_cfg.get("d", 2)),
            kernel_length=int(model_cfg.get("kernel_length", 64)),
            dropout=float(model_cfg.get("dropout", 0.25)),
            max_norm_depthwise=float(model_cfg.get("max_norm_depthwise", 1.0)),
            max_norm_linear=float(model_cfg.get("max_norm_linear", 0.25)),
        )
    if key == "atcnet_tf_faithful":
        return ATCNetOfficialFaithful(
            n_ch=n_ch,
            n_t=n_t,
            n_classes=n_classes,
            n_windows=int(model_cfg.get("n_windows", 5)),
            eegn_f1=int(model_cfg.get("eegn_f1", 16)),
            eegn_d=int(model_cfg.get("eegn_d", 2)),
            eegn_kernel_size=int(model_cfg.get("eegn_kernel_size", 64)),
            eegn_pool_size=int(model_cfg.get("eegn_pool_size", 7)),
            eegn_dropout=float(model_cfg.get("eegn_dropout", 0.3)),
            tcn_depth=int(model_cfg.get("tcn_depth", 2)),
            tcn_kernel_size=int(model_cfg.get("tcn_kernel_size", 4)),
            tcn_filters=int(model_cfg.get("tcn_filters", 32)),
            tcn_dropout=float(model_cfg.get("tcn_dropout", 0.3)),
            conv_weight_decay=float(model_cfg.get("conv_weight_decay", 0.009)),
            dense_weight_decay=float(model_cfg.get("dense_weight_decay", 0.5)),
            conv_max_norm=float(model_cfg.get("conv_max_norm", 0.6)),
        )
    raise ValueError(f"Unsupported faithful model key: {model_key}")
