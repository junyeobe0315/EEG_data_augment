from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.augment.generate import (
    compute_target_counts,
    build_synthetic_with_qc,
    build_synthetic_pool,
    select_from_pool,
    select_indices_from_pool,
)
from src.data.dataset import load_index, load_samples
from src.data.normalize import ZScoreNormalizer
from src.data.subsample import load_split_indices
from src.eval.distance import classwise_distance_summary
from src.eval.metrics import compute_metrics
from src.eval.embedding import FrozenEEGNetEmbedder, train_embedding_eegnet
from src.qc.qc_pipeline import fit_qc
from src.train.train_classifier import train_classifier
from src.train.train_generator import train_generator, LoadedGeneratorSampler
from src.train.proxy_select_ckpt import select_best_checkpoint
from src.utils.alpha import alpha_ratio_to_mix
from src.utils.config import config_hash, save_yaml
from src.utils.io import ensure_dir, write_json
from src.utils.resource import get_git_commit
from src.utils.results import make_run_id
from src.utils.seed import set_global_seed, stable_hash_seed


def _resolve_device(train_cfg: dict) -> str:
    """Resolve training device string from config.

    Inputs:
    - train_cfg: training config dict with "device" key.

    Outputs:
    - device string ("cuda" or "cpu" or explicit device).

    Internal logic:
    - Uses CUDA when available if device is "auto".
    """
    req = str(train_cfg.get("device", "auto"))
    if req == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return req


def _resolve_batch_size(train_cfg: dict, r: float) -> int:
    """Resolve batch size with optional r-dependent logic.

    Supported (optional) keys in train_cfg:
    - batch_size_by_r: mapping of {r_value: batch_size}, chooses max r_value <= r.
    - batch_size_scale_with_r: bool, scales base batch_size by r / r_ref.
      r_ref defaults to 0.1, min_scale defaults to 1.0, max_scale optional.

    Inputs:
    - train_cfg: training config dict.
    - r: low-data fraction.

    Outputs:
    - resolved batch size (int).

    Internal logic:
    - Applies explicit r-to-batch mapping first, otherwise scales base size.
    """
    base = int(train_cfg.get("batch_size", 32))
    by_r = train_cfg.get("batch_size_by_r")
    if isinstance(by_r, dict) and by_r:
        items = sorted((float(k), int(v)) for k, v in by_r.items())
        chosen = items[0][1]
        for rk, bs in items:
            if r >= rk:
                chosen = bs
        return max(1, int(chosen))
    if bool(train_cfg.get("batch_size_scale_with_r", False)):
        r_ref = float(train_cfg.get("batch_size_r_ref", 0.1))
        min_scale = float(train_cfg.get("batch_size_min_scale", 1.0))
        max_scale = train_cfg.get("batch_size_max_scale")
        scale = max(min_scale, float(r) / max(r_ref, 1e-6))
        if max_scale is not None:
            scale = min(scale, float(max_scale))
        return max(1, int(round(base * scale)))
    return base


def _generator_run_id(
    subject: int,
    seed: int,
    r: float,
    generator: str,
    dataset_cfg: dict,
    preprocess_cfg: dict,
    gen_cfg: dict,
) -> str:
    """Build a deterministic generator cache ID from configs.

    Inputs:
    - subject/seed/r/generator: identifiers for the run.
    - dataset_cfg/preprocess_cfg/gen_cfg: config dicts that affect generator output.

    Outputs:
    - short hash string for generator cache directory.

    Internal logic:
    - Hashes a payload of relevant settings to ensure cache validity.
    """
    payload = {
        "dataset": dataset_cfg.get("name", "bci2a"),
        "subject": subject,
        "seed": seed,
        "r": r,
        "generator": generator,
        "gen_model": gen_cfg.get("model", {}),
        "gen_train": gen_cfg.get("train", {}),
        "preprocess": preprocess_cfg,
    }
    return config_hash(payload)


def _list_ckpts(gen_run_dir: Path) -> list[str]:
    """List generator checkpoint files in a run directory.

    Inputs:
    - gen_run_dir: path to generator run directory.

    Outputs:
    - sorted list of checkpoint paths as strings.

    Internal logic:
    - Globs ckpt_epoch_*.pt and sorts lexicographically.
    """
    ckpts = sorted(gen_run_dir.glob("ckpt_epoch_*.pt"))
    return [str(p) for p in ckpts]


def _torch_linear_probe_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    num_classes: int,
    cfg: dict,
    device: str,
) -> np.ndarray:
    """Train a lightweight linear probe on embeddings and predict validation labels."""
    max_iter = int(cfg.get("max_iter", 200))
    lr = float(cfg.get("lr", 1e-2))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    batch_size = int(cfg.get("batch_size", 512))

    x_train_t = torch.from_numpy(x_train.astype(np.float32))
    y_train_t = torch.from_numpy(y_train.astype(np.int64))
    x_val_t = torch.from_numpy(x_val.astype(np.float32))

    model = torch.nn.Linear(int(x_train.shape[1]), int(num_classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    n = int(x_train_t.shape[0])
    if n <= 0:
        return np.zeros((0,), dtype=np.int64)

    model.train()
    for _ in range(max_iter):
        if batch_size >= n:
            xb = x_train_t.to(device)
            yb = y_train_t.to(device)
        else:
            idx = torch.randint(0, n, (batch_size,))
            xb = x_train_t[idx].to(device)
            yb = y_train_t[idx].to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred = torch.argmax(model(x_val_t.to(device)), dim=1)
    return pred.detach().cpu().numpy().astype(np.int64)


def run_experiment(
    subject: int,
    seed: int,
    r: float,
    method: str,
    classifier: str,
    generator: str,
    alpha_ratio: float,
    qc_on: bool,
    dataset_cfg: dict,
    preprocess_cfg: dict,
    split_cfg: dict,
    model_cfgs: dict,
    gen_cfgs: dict,
    qc_cfg: dict,
    results_path: str | Path,
    run_root: str | Path = "./artifacts/runs",
    stage: str = "full",
    compute_distance: bool = True,
    alpha_search_cfg: dict | None = None,
    config_pack: str = "base",
) -> dict[str, Any]:
    """Run a single experiment configuration end-to-end.

    Inputs:
    - subject/seed/r: dataset split identifiers.
    - method/classifier/generator/alpha_ratio/qc_on: treatment knobs.
    - dataset_cfg/preprocess_cfg/split_cfg/model_cfgs/gen_cfgs/qc_cfg: configs.
    - results_path: path to results.csv (for consistency, though row is returned).
    - run_root: root directory for run artifacts.
    - stage: "alpha_search" | "final_eval" | "full".
    - compute_distance: whether to compute embedding-based distances.
    - alpha_search_cfg: optional config for alpha-search proxies.

    Outputs:
    - row dict containing metrics, diagnostics, and metadata for results.csv.

    Internal logic:
    - Loads cached data and split indices, fits normalizer on train_sub only.
    - Runs GenAug branch (generator, QC, sampling, optional distance) or baseline.
    - Trains classifier (or linear-probe proxy) and assembles a results row.
    """
    start_time = time.time()
    set_global_seed(seed)

    index_df = load_index(dataset_cfg["index_path"])  # sample metadata table
    splits = load_split_indices(dataset_cfg["name"], subject, seed, r, root="./artifacts/splits")  # split indices

    x_train, y_train = load_samples(index_df, splits["train_sub"])  # T_train_subsample
    x_val, y_val = load_samples(index_df, splits["val"])  # T_val
    x_test, y_test = load_samples(index_df, splits["test"])  # E_test

    normalizer = ZScoreNormalizer(eps=float(preprocess_cfg.get("normalization", {}).get("eps", 1.0e-6)))
    normalizer.fit(x_train)  # fit on T_train_subsample only (leak-free)
    x_train = normalizer.transform(x_train)
    x_val = normalizer.transform(x_val)
    x_test = normalizer.transform(x_test)
    normalizer_state = normalizer.state_dict()  # saved for embedding + checkpoints

    num_classes = int(np.max(y_train)) + 1  # inferred class count
    run_key = {
        "config_pack": str(config_pack),
        "subject": subject,
        "seed": seed,
        "r": r,
        "classifier": classifier,
        "method": method,
        "generator": generator,
        "qc_on": bool(qc_on),
        "alpha_ratio": float(alpha_ratio),
    }

    run_id = make_run_id(run_key)  # deterministic run id for reproducibility
    run_dir = ensure_dir(Path(run_root) / run_id)  # per-run artifact directory
    save_yaml(run_dir / "config_snapshot.yaml", {
        "dataset": dataset_cfg,
        "preprocess": preprocess_cfg,
        "split": split_cfg,
        "models": model_cfgs,
        "generators": gen_cfgs,
        "qc": qc_cfg,
        "config_pack": str(config_pack),
        "run": run_key,
        "stage": stage,
    })

    synth_data = None
    qc_report = {"pass_rate": np.nan, "oversample_factor": np.nan}
    distance_rows: dict[str, Any] = {}
    ratio_effective = 0.0
    alpha_mix_effective = 0.0
    pool_enabled = False  # default: no pool unless GenAug builds it
    pool_alpha = 0.0  # default pool ratio
    pool_x = None  # cached pool samples (if built)
    pool_y = None  # cached pool labels (if built)
    target_counts: dict[int, int] = {}  # per-class synth targets
    gen_run_dir: Path | None = None  # generator run dir if built

    if method == "GenAug":
        if alpha_ratio <= 0.0:
            synth_data = (np.empty((0,) + x_train.shape[1:], dtype=np.float32), np.empty((0,), dtype=np.int64))
            ratio_effective = 0.0
            alpha_mix_effective = 0.0
        else:
            gen_cfg = gen_cfgs[generator]
            gen_id = _generator_run_id(
                subject=subject,
                seed=seed,
                r=r,
                generator=generator,
                dataset_cfg=dataset_cfg,
                preprocess_cfg=preprocess_cfg,
                gen_cfg=gen_cfg,
            )
            gen_run_dir = ensure_dir(Path("./artifacts/checkpoints") / f"gen_{gen_id}")
            gen_seed = stable_hash_seed(seed, {"generator": generator, "subject": subject, "r": r})

            ckpts = _list_ckpts(gen_run_dir)
            if not ckpts:
                gen_train_cfg = dict(gen_cfg["train"])
                gen_train_cfg["batch_size"] = _resolve_batch_size(gen_train_cfg, r)
                gen_train = train_generator(
                    x_train=x_train,
                    y_train=y_train,
                    model_type=generator,
                    model_cfg=gen_cfg["model"],
                    train_cfg=gen_train_cfg,
                    run_dir=gen_run_dir,
                    seed=gen_seed,
                )
                ckpts = gen_train.get("ckpts", [])
                if not ckpts:
                    ckpts = _list_ckpts(gen_run_dir)

            if not ckpts:
                raise RuntimeError("No generator checkpoints produced.")

            ckpt_sel_cfg = gen_cfg.get("checkpoint_selection", {})
            if ckpt_sel_cfg.get("enabled", True):
                best_ckpt_path = gen_run_dir / "best_ckpt.json"
                if best_ckpt_path.exists():
                    try:
                        import json

                        best_ckpt = json.loads(best_ckpt_path.read_text()).get("best_ckpt")
                    except Exception:
                        best_ckpt = None
                    if not best_ckpt:
                        best_ckpt = ckpts[-1]
                else:
                    sel = select_best_checkpoint(
                        ckpt_paths=ckpts,
                        x_train=x_train,
                        y_train=y_train,
                        x_val=x_val,
                        y_val=y_val,
                        gen_cfg=gen_cfg,
                        proxy_model_cfg=model_cfgs["eegnet"]["model"],
                        qc_cfg=qc_cfg,
                        num_classes=num_classes,
                        run_dir=gen_run_dir,
                        seed=gen_seed,
                    )
                    best_ckpt = sel.get("best_ckpt") or ckpts[-1]
            else:
                best_ckpt = ckpts[-1]

            target_counts = compute_target_counts(y_train, alpha_ratio)  # per-class synth targets
            qc_state = None
            if qc_on:
                qc_state = fit_qc(x_train, y_train, sfreq=int(dataset_cfg.get("sfreq", 250)), cfg=qc_cfg)

            sample_cfg = gen_cfg.get("sample", {})
            buffer = float(sample_cfg.get("dynamic_buffer", 1.2))  # oversample buffer
            ddpm_steps = sample_cfg.get("ddpm_steps")  # optional DDPM steps
            sample_device = _resolve_device(gen_cfg.get("train", {}))
            sampler = LoadedGeneratorSampler(best_ckpt, device=sample_device)

            def _sample_fn(cls: int, n: int) -> np.ndarray:
                """Sample n synthetic trials for a given class.

                Inputs:
                - cls: class index to condition on.
                - n: number of samples to generate.

                Outputs:
                - ndarray [n, C, T] synthetic samples.

                Internal logic:
                - Builds a label vector and calls the generator sampler.
                """
                y = np.full((n,), int(cls), dtype=np.int64)
                return sampler.sample(y, ddpm_steps=ddpm_steps)

            pool_cfg = gen_cfg.get("pool", {})
            pool_enabled = bool(pool_cfg.get("enabled", False))  # enable pooled sampling
            pool_alpha = float(pool_cfg.get("alpha_ratio_max", alpha_ratio))  # pool target ratio
            if pool_alpha < alpha_ratio:
                pool_alpha = float(alpha_ratio)

            x_syn = np.empty((0,))  # synthetic samples (init)
            y_syn = np.empty((0,), dtype=np.int64)  # synthetic labels (init)
            pool_report: dict[str, Any] = {}
            select_report: dict[str, Any] = {}
            pool_x = None
            pool_y = None

            if pool_enabled and pool_alpha > 0.0:
                pool_tag = f"pool_alpha_{pool_alpha:g}_qc_{int(qc_on)}"
                pool_npz = gen_run_dir / f"{pool_tag}.npz"
                pool_meta = gen_run_dir / f"{pool_tag}.json"

                if pool_npz.exists():
                    arr = np.load(pool_npz)
                    pool_x = arr["X"]
                    pool_y = arr["y"]
                    if pool_meta.exists():
                        try:
                            import json

                            pool_report = json.loads(pool_meta.read_text())
                        except Exception:
                            pool_report = {}
                else:
                    target_counts_pool = compute_target_counts(y_train, pool_alpha)
                    pool_x, pool_y, pool_report = build_synthetic_pool(
                        sample_fn=_sample_fn,
                        target_counts=target_counts_pool,
                        qc_state=qc_state if qc_on else None,
                        qc_cfg=qc_cfg,
                        sfreq=int(dataset_cfg.get("sfreq", 250)),
                        buffer=buffer,
                    )
                    np.savez_compressed(pool_npz, X=pool_x, y=pool_y)
                    try:
                        write_json(pool_meta, pool_report)
                    except Exception:
                        pass

                if pool_x is None or pool_y is None:
                    raise RuntimeError("Pool cache is invalid.")
            else:
                x_syn, y_syn, qc_report = build_synthetic_with_qc(
                    sample_fn=_sample_fn,
                    target_counts=target_counts,
                    qc_state=qc_state if qc_on else None,
                    qc_cfg=qc_cfg,
                    sfreq=int(dataset_cfg.get("sfreq", 250)),
                    buffer=buffer,
                )
            if pool_enabled and pool_alpha > 0.0:
                select_seed = stable_hash_seed(seed, {"alpha_ratio": alpha_ratio, "pool_tag": pool_tag})
                x_syn, y_syn, select_report = select_from_pool(
                    x_pool=pool_x,
                    y_pool=pool_y,
                    target_counts=target_counts,
                    seed=select_seed,
                )
                qc_report = dict(pool_report)
                qc_report.update(select_report)

            synth_data = (x_syn, y_syn)
            ratio_effective = float(len(y_syn) / max(1, len(x_train)))
            alpha_mix_effective = alpha_ratio_to_mix(ratio_effective)

            # Distance analysis
            if compute_distance and len(x_syn) > 0:
                embed_dir = ensure_dir(run_dir / "embedding")
                embed_ckpt = embed_dir / "ckpt.pt"
                if not embed_ckpt.exists():
                    embed_ckpt = train_embedding_eegnet(
                        x_train=x_train,
                        y_train=y_train,
                        x_val=x_val,
                        y_val=y_val,
                        model_cfg=model_cfgs["eegnet"]["model"],
                        train_cfg=model_cfgs["eegnet"]["train"],
                        run_dir=embed_dir,
                        normalizer_state=normalizer_state,
                        num_classes=num_classes,
                    )
                embedder = FrozenEEGNetEmbedder(embed_ckpt, device=_resolve_device(model_cfgs["eegnet"]["train"]))
                real_emb = embedder.transform(x_train)
                syn_emb = embedder.transform(x_syn)
                distance_rows = classwise_distance_summary(
                    real_emb,
                    y_train,
                    syn_emb,
                    y_syn,
                    num_classes=num_classes,
                    n_projections=int(gen_cfg.get("distance", {}).get("n_projections", 64)),
                    seed=seed,
                )

    model_cfg = model_cfgs[classifier]["model"]
    train_cfg = dict(model_cfgs[classifier]["train"])
    train_cfg["batch_size"] = _resolve_batch_size(train_cfg, r)
    eval_cfg = model_cfgs[classifier]["evaluation"]

    alpha_search_cfg = alpha_search_cfg or {}
    proxy_mode = str(alpha_search_cfg.get("proxy_mode", "full"))
    use_linear_probe = (
        stage == "alpha_search"
        and proxy_mode == "linear_probe"
        and method == "GenAug"
        and classifier == "eegnet"
    )

    if use_linear_probe:
        gen_id_for_embed = _generator_run_id(subject, seed, r, generator, dataset_cfg, preprocess_cfg, gen_cfgs[generator])
        embed_dir = ensure_dir(Path("./artifacts/checkpoints") / f"gen_{gen_id_for_embed}" / "embedder_proxy")
        embed_ckpt = embed_dir / "ckpt.pt"
        if not embed_ckpt.exists():
            embed_train_cfg = dict(model_cfgs["eegnet"]["train"])
            embed_train_cfg["batch_size"] = _resolve_batch_size(embed_train_cfg, r)
            embed_steps = alpha_search_cfg.get("embed_steps")
            if embed_steps is not None:
                embed_train_cfg["step_control"] = {
                    "enabled": True,
                    "total_steps": int(embed_steps),
                    "steps_per_eval": max(1, int(embed_steps) // 3),
                }
                embed_train_cfg["epochs"] = 1
            embed_ckpt = train_embedding_eegnet(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                model_cfg=model_cfgs["eegnet"]["model"],
                train_cfg=embed_train_cfg,
                run_dir=embed_dir,
                normalizer_state=normalizer_state,
                num_classes=num_classes,
            )

        embedder = FrozenEEGNetEmbedder(embed_ckpt, device=_resolve_device(model_cfgs["eegnet"]["train"]))
        train_emb_path = embed_dir / "train_emb.npz"
        val_emb_path = embed_dir / "val_emb.npz"

        if train_emb_path.exists():
            train_emb = np.load(train_emb_path)["Z"]
        else:
            train_emb = embedder.transform(x_train)
            np.savez_compressed(train_emb_path, Z=train_emb)

        if val_emb_path.exists():
            val_emb = np.load(val_emb_path)["Z"]
        else:
            val_emb = embedder.transform(x_val)
            np.savez_compressed(val_emb_path, Z=val_emb)

        syn_emb = None
        syn_y = None
        if (
            pool_enabled
            and pool_alpha > 0.0
            and pool_x is not None
            and pool_y is not None
            and len(pool_x) > 0
        ):
            pool_tag = f"pool_alpha_{pool_alpha:g}_qc_{int(qc_on)}"
            pool_emb_path = gen_run_dir / f"{pool_tag}_emb.npz"
            if pool_emb_path.exists():
                arr = np.load(pool_emb_path)
                pool_emb = arr["Z"]
                pool_y = arr["y"]
            else:
                pool_emb = embedder.transform(pool_x)
                np.savez_compressed(pool_emb_path, Z=pool_emb, y=pool_y)

            select_seed = stable_hash_seed(seed, {"alpha_ratio": alpha_ratio, "pool_tag": pool_tag})
            indices, _ = select_indices_from_pool(pool_y, target_counts, seed=select_seed)
            if len(indices) > 0:
                syn_emb = pool_emb[indices]
                syn_y = pool_y[indices]
                ratio_effective = float(len(indices) / max(1, len(x_train)))
                alpha_mix_effective = alpha_ratio_to_mix(ratio_effective)
        elif synth_data is not None and len(synth_data[0]) > 0:
            syn_emb = embedder.transform(synth_data[0])
            syn_y = synth_data[1]

        if syn_emb is not None and syn_y is not None and len(syn_emb) > 0:
            x_emb = np.concatenate([train_emb, syn_emb], axis=0)
            y_emb = np.concatenate([y_train, syn_y], axis=0)
        else:
            x_emb = train_emb
            y_emb = y_train

        lp_cfg = alpha_search_cfg.get("linear_probe", {})
        lp_backend = str(lp_cfg.get("backend", "torch")).lower()
        if lp_backend == "sklearn":
            from sklearn.linear_model import LogisticRegression

            lr_kwargs = {
                "max_iter": int(lp_cfg.get("max_iter", 200)),
                "C": float(lp_cfg.get("C", 1.0)),
                "solver": str(lp_cfg.get("solver", "lbfgs")),
            }
            if "n_jobs" in lp_cfg:
                lr_kwargs["n_jobs"] = int(lp_cfg.get("n_jobs", 1))
            clf = LogisticRegression(**lr_kwargs)
            clf.fit(x_emb, y_emb)
            pred = clf.predict(val_emb)
        else:
            pred = _torch_linear_probe_predict(
                x_train=x_emb,
                y_train=y_emb,
                x_val=val_emb,
                num_classes=num_classes,
                cfg=lp_cfg,
                device=_resolve_device(model_cfgs["eegnet"]["train"]),
            )
        val_metrics = compute_metrics(y_val, pred)
        metrics = {
            "val_acc": val_metrics["acc"],
            "val_bal_acc": val_metrics["bal_acc"],
            "val_kappa": val_metrics["kappa"],
            "val_macro_f1": val_metrics["macro_f1"],
            "acc": np.nan,
            "bal_acc": np.nan,
            "kappa": np.nan,
            "macro_f1": np.nan,
            "ratio_effective": ratio_effective,
            "alpha_mix_effective": alpha_mix_effective,
        }
    else:
        evaluate_test = stage in {"final_eval", "full"}
        metrics = train_classifier(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            model_type=classifier,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            eval_cfg=eval_cfg,
            method=method,
            alpha_ratio=alpha_ratio,
            num_classes=num_classes,
            run_dir=run_dir,
            normalizer_state=normalizer_state,
            synth_data=synth_data,
            evaluate_test=evaluate_test,
            aug_cfg=model_cfgs.get("augment", {}),
        )

    runtime_sec = float(time.time() - start_time)
    run_device = _resolve_device(train_cfg)
    if run_device.startswith("cuda") and torch.cuda.is_available():
        gpu_idx = torch.device(run_device).index
        if gpu_idx is None:
            gpu_idx = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_idx)
        gpu_mem_mb = float(torch.cuda.max_memory_allocated(gpu_idx) / (1024.0 ** 2))
    else:
        gpu_name = np.nan
        gpu_mem_mb = np.nan

    distance_value = distance_rows.get("dist_mmd", distance_rows.get("dist_swd", np.nan))
    row = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": get_git_commit(),
        "config_hash": config_hash({
            "dataset": dataset_cfg,
            "preprocess": preprocess_cfg,
            "split": split_cfg,
            "models": model_cfgs,
            "generators": gen_cfgs,
            "qc": qc_cfg,
        }),
        "dataset": dataset_cfg["name"],
        "config_pack": str(config_pack),
        "subject": subject,
        "seed": seed,
        "r": r,
        "method": method,
        "classifier": classifier,
        "generator": generator,
        "alpha_ratio": float(alpha_ratio),
        "qc_on": bool(qc_on),
        "acc": metrics.get("acc", np.nan),
        "bal_acc": metrics.get("bal_acc", np.nan),
        "kappa": metrics.get("kappa", np.nan),
        "macro_f1": metrics.get("macro_f1", np.nan),
        "val_acc": metrics.get("val_acc", np.nan),
        "val_bal_acc": metrics.get("val_bal_acc", np.nan),
        "val_kappa": metrics.get("val_kappa", np.nan),
        "val_macro_f1": metrics.get("val_macro_f1", np.nan),
        "pass_rate": qc_report.get("pass_rate", np.nan),
        "oversample_factor": qc_report.get("oversample_factor", np.nan),
        "distance": distance_value,
        "ratio_effective": ratio_effective,
        "alpha_mix_effective": alpha_mix_effective,
        "runtime_sec": runtime_sec,
        "device": run_device,
        "gpu_name": gpu_name,
        "gpu_mem_mb": gpu_mem_mb,
    }
    row.update(distance_rows)

    write_json(run_dir / "metrics_row.json", row)
    return row
