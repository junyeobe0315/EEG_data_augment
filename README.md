# EEG Class-Conditional GenAug (BCI IV 2a)

This repository provides a **leak-free, reproducible** pipeline to evaluate
class-conditional generative augmentation for EEG classification under
cross-session and low-data conditions.

Key goals:
- Verify if GenAug improves performance under low-data regimes.
- Validate the bias–variance (approximation–estimation) trade-off in practice.
- Link performance gains to class-conditional distribution mismatch in an embedding space.

Legacy code is preserved in `legacy/`.

## Quick Start

1) Cache preprocessed data and build index:
```bash
python scripts/cache_preprocess.py
```

2) Prepare split indices:
```bash
python scripts/prepare_splits.py
```

3) Run a single baseline:
```bash
python scripts/run_single.py --subject 1 --seed 0 --r 0.1 --method C0 --classifier eegnet
```

4) Stage-1 alpha search (EEGNet only):
```bash
python scripts/run_grid.py --stage alpha_search
python scripts/select_alpha.py --metric val_kappa
```

5) Stage-2 final evaluation:
```bash
python scripts/run_grid.py --stage final_eval
```

6) Summaries / plots:
```bash
python scripts/summarize_results.py
python scripts/plot_results.py --metric kappa
```

## Directory Layout
```
configs/               # All YAML configs
scripts/               # Entry-point scripts
src/                   # Core modules
tests/                 # Leakage/shape/schema tests
artifacts/             # Splits, runs, checkpoints, figures
results/               # results.csv and summaries
legacy/                # Previous codebase
```

## Reproducibility Rules (Enforced)
- Split files are the source of truth.
- All `fit()` steps use **T_train_subsample only**.
- Results are appended to one `results/results.csv`.
- Resume-safe runs: if primary key exists, the run is skipped.

## Notes
- Generators are trained only on the low-data training subset.
- Embedding φ for distance metrics is trained only on real training data.
- Distance metrics (MMD/SWD) are logged per class and aggregated by class priors.
