# Main Experimental Protocol (BCI IV-2a)

This document defines the **main leakage-safe protocol** used for augmentation claims.

## 1) Objective
Evaluate whether generative augmentation improves cross-session MI decoding when training data is limited.

- Train domain: session `T`
- Target/test domain: session `E`
- Target risk: final performance on `P_E` (cross-session setting)

## 2) Split Definition
For each subject (`A01`-`A09`):
- `T_train` / `T_val`: stratified split inside session `T` (default 80/20)
- `E_test`: full session `E`
- All split/subsample indices are saved under `data/splits/` and reused.

Note:
- In `cross_session`, `split.test_ratio` is ignored by design (`E` is always full test).

## 3) Low-Data Setting
Low-data fraction `r` is applied to **`T_train` only**.

Default: `r in {0.01, 0.05, 0.10, 0.20, 1.00}` (`1.00` is the full-data reference)

- `T_val` remains full.
- `E_test` remains full.
- Subsampling is class-stratified and seed-controlled.

## 4) Augmentation Parameterization
The sweep parameter is:
- `rho = N_synth / N_real`

If mixture-weight interpretation is needed:
- `alpha_tilde = rho / (1 + rho)`

Default ratio grid:
- `rho in {0.0, 0.25, 0.5, 1.0, 2.0}`

Conditions:
- `C0_no_aug`
- `C1_classical`
- `C2_hard_mix` (legacy hard-label mix baseline)
- `GenAug_{generator}` with `QC OFF/ON`

## 5) Leakage Rules (Non-negotiable)
Fit on `T_train` only:
- normalization/scaling statistics
- FBCSP extractor
- generator training
- QC statistics/thresholds

Selection/evaluation:
- `T_val`: early stopping and hyperparameter/model selection only
- `E_test`: final report only

## 6) Generator Checkpoint Selection
Generator checkpoints are selected with deterministic `T_val` proxy:
1. save periodic checkpoints (`runs/gen/.../checkpoints/epoch_*.pt`)
2. for each checkpoint, sample class-balanced synthetic batch
3. apply QC fitted from train-only real samples
4. run short proxy classifier training and evaluate on `T_val`
5. select best checkpoint by proxy metric (default: balanced accuracy)

Artifacts:
- `runs/gen/.../ckpt_list.json`
- `runs/gen/.../ckpt_scores.json`
- `runs/gen/.../training_meta.json`
- selected checkpoint copy: `runs/gen/.../ckpt.pt`

## 7) Determinism and Logging
- Global seed control for Python/NumPy/Torch/CUDA.
- Classifier runs use condition-stable seeds (`seed + hash(split, model, mode, ratio, generator, qc)`), so partial reruns do not change random augmentation behavior.
- Sampling/QC deterministic seeds are logged (`synth_seed`, `qc_seed`).
- Synthetic metadata includes generator ckpt path/hash and sampling params.

## 8) Step-Matching Control
To prevent confound from different optimizer-update counts:
- Deep classifiers use fixed-step control (`train.step_control`).
- Log includes `total_steps_target`, `total_steps_done`, and train sizes.

## 9) Test-Final-Only Workflow
Default main sweep runs with:
- `configs/clf.yaml -> evaluation.evaluate_test: false`

Workflow:
1. run sweep and select settings on `T_val`
2. run final test-only evaluation once using saved checkpoints

## 10) Statistical Reporting
Minimum report set:
- per-subject, per-seed: `acc`, `kappa`, `macro-F1` (and optional `bal_acc`)
- aggregate: mean ± std/CI
- paired subject-level comparison vs baseline:
  - paired t-test
  - Wilcoxon signed-rank
  - effect sizes (paired Cohen's d, rank-biserial)

## 11) Commands
```bash
# Main 00-05
CONDA_ENV=EEG ./scripts/run_main_pipeline.sh

# Final test-only pass
conda run -n EEG --no-capture-output python scripts/05b_final_test_eval.py \
  --input-csv results/metrics/clf_cross_session.csv \
  --output-csv results/metrics/clf_cross_session_test.csv

# Aggregate/figures
conda run -n EEG --no-capture-output python scripts/05_eval_and_aggregate.py \
  --metrics-file clf_cross_session_test.csv
```

## 12) Key Outputs
- `results/metrics/clf_cross_session.csv`
- `results/metrics/clf_cross_session_test.csv`
- `results/tables/main_table_{acc|kappa|f1_macro}.csv`
- `results/tables/stats_summary.csv`
- `results/tables/distance_gain_correlation.csv`
- `results/figures/accuracy_vs_ratio.png`
- `results/figures/accuracy_vs_r.png`
- `results/figures/distance_vs_gain.png`
