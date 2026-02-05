from __future__ import annotations

import numpy as np
import torch

from src.models.classifiers.eegnet import EEGNet
from src.models.classifiers.eegconformer import EEGConformer
from src.models.classifiers.ctnet import CTNet
from src.models.classifiers.fbcsp_svm import SVMClassifier


def test_deep_model_shapes() -> None:
    x = torch.randn(4, 22, 1000)
    n_classes = 4

    model = EEGNet(n_ch=22, n_t=1000, n_classes=n_classes)
    y = model(x)
    assert y.shape == (4, n_classes)

    model = EEGConformer(n_ch=22, n_t=1000, n_classes=n_classes)
    y = model(x)
    assert y.shape == (4, n_classes)

    model = CTNet(n_ch=22, n_t=1000, n_classes=n_classes)
    y = model(x)
    assert y.shape == (4, n_classes)


def test_fbcsp_svm_shape() -> None:
    x = np.random.randn(20, 22, 500).astype(np.float32)
    y = np.random.randint(0, 4, size=20)
    clf = SVMClassifier()
    clf.fit(x, y)
    pred = clf.predict(x)
    assert pred.shape == (20,)
