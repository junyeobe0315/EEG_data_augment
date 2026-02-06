from __future__ import annotations

import numpy as np

from src.train.train_classifier import train_classifier


def test_svm_mixup_path_runs(tmp_path) -> None:
    """SVM+C2 path should run without referencing undefined mixup settings."""
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(24, 6, 128)).astype(np.float32)
    y_train = np.tile(np.arange(4, dtype=np.int64), 6)
    x_val = rng.normal(size=(8, 6, 128)).astype(np.float32)
    y_val = np.tile(np.arange(4, dtype=np.int64), 2)
    x_test = rng.normal(size=(8, 6, 128)).astype(np.float32)
    y_test = np.tile(np.arange(4, dtype=np.int64), 2)

    metrics = train_classifier(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        model_type="svm",
        model_cfg={
            "C": 1.0,
            "kernel": "linear",
            "gamma": "scale",
            "sfreq": 250,
            "n_components": 2,
            "bands": [[8, 12]],
        },
        train_cfg={"mixup_alpha": 0.2},
        eval_cfg={"best_metric": "kappa", "best_direction": "max"},
        method="C2",
        alpha_ratio=0.0,
        num_classes=4,
        run_dir=tmp_path / "svm_mixup",
        normalizer_state={},
        synth_data=None,
        evaluate_test=False,
        aug_cfg={},
    )

    assert "val_kappa" in metrics
    assert "val_bal_acc" in metrics
