# Cross-Session Low-Data Protocol (BCI IV-2a)

## 1) Goal
Validate whether generative EEG augmentation improves cross-session generalization under low-data conditions, with strict leakage control and reproducibility.

Theory-facing definition for this project:
- We sweep `rho = N_synth / N_real` (synthetic:real ratio), not a direct mixture weight.
- When a mixture-weight interpretation is needed, we convert with:
  - `alpha_tilde = rho / (1 + rho)` in `[0, 1)`.

## 2) Fixed Split
- Subject: A01~A09
- Session split (cross-session):
  - Train domain: `AxxT`
  - Test domain: `AxxE`
- Inside `AxxT`:
  - `T_train`: 80% stratified
  - `T_val`: 20% stratified
- `AxxE` is `E_test` (never used for model selection)
- Split/subsample indices are saved and reused from `data/splits/*.json`.

## 3) Low-Data Definition
- Low-data ratio `r` is applied **only** to `T_train`.
- Default `r` grid: `{0.01, 0.05, 0.10, 0.20}`.
- Subsample is class-stratified and saved by `(subject, seed, r)`.
- `T_val` and `E_test` remain full.

## 3.1) Cross-Session Risk Statement
- Let `P_T` be the train-session (`T`) distribution and `P_E` be the test-session (`E`) distribution.
- The target objective is final risk on `P_E` (not same-distribution risk on `P_T`).
- Working hypothesis: under low-data, augmentation reduces estimator variance and stabilizes representation learning on `P_T`; this stability can transfer to `P_E` under moderate domain shift.

## 4) Leakage Rules (Non-negotiable)
- Fit-only-on-train for:
  - normalization/scaling
  - FBCSP features
  - generator training
  - QC statistics/threshold references
- `T_val` usage:
  - early stopping / hyperparameter selection only
- `E_test` usage:
  - final report only (no tuning)
- Repeated runs with seeds (`0..4` by default) and report mean±std/CI.

## 5) Augmentation Grid
- Ratio `rho`: `{0.0, 0.25, 0.5, 1.0, 2.0}`
- Conditions:
  - `C0_no_aug`
  - `C1_classical` (time-shift + jitter + channel dropout)
  - `C2_mixup`
  - `GenAug_{generator}` with QC on/off
- For all augmentation conditions, dataset expansion is controlled by `rho`.
- Derived reporting column: `alpha_tilde = rho / (1 + rho)`.

## 6) Distance-vs-Gain Analysis
- Embedding `phi(x)`: frozen EEGNet baseline checkpoint per split (trained on real-only `C0_no_aug`).
- Distances per class:
  - Sliced Wasserstein
  - MMD (RBF)
- Distances are computed on pushforward class-conditional distributions:
  - `phi#P_y` vs `phi#Q_y`
- Analyze relation between distance and accuracy gain (`gain_acc`) over runs.
- Outputs include scatter plot and correlation/regression tables.

## 6.1) Step-Matching Control (Confound Prevention)
- To prevent “augmentation helps only because optimization runs longer”, deep classifier training uses fixed-step control:
  - `train.step_control.enabled: true`
  - `train.step_control.total_steps: <fixed>`
  - `train.step_control.steps_per_eval: <fixed>`
- This makes total optimizer updates comparable across conditions.
- Each run logs:
  - `total_steps_target`, `total_steps_done`, `n_train_real`, `n_train_aug`, `batch_size`, `sampling_strategy`.

## 7) Run Commands
```bash
# Full pipeline (00~05)
CONDA_ENV=EEG ./scripts/run_main_pipeline.sh

# Individual steps
python scripts/00_prepare_data.py
python scripts/01_make_splits.py
python scripts/02_train_gen.py
python scripts/03_sample_and_qc.py
python scripts/04_train_clf.py
python scripts/05_eval_and_aggregate.py
```

## 8) Key Outputs
- Per-run metrics: `results/metrics/clf_cross_session.csv`
- Aggregated tables: `results/tables/main_table_{acc|kappa|f1_macro}.csv`
- Reproducibility gate: `results/tables/reproducibility_check.csv`
- Distance-gain stats: `results/tables/distance_gain_correlation.csv`
- Figures:
  - `results/figures/accuracy_vs_ratio.png`
  - `results/figures/accuracy_vs_alpha.png` (legacy alias)
  - `results/figures/accuracy_vs_r.png`
  - `results/figures/distance_vs_gain.png`

---

# 한국어 요약

- 고정 프로토콜: `cross-session` (`T`로 학습/검증, `E`로 최종 테스트)
- low-data는 `T_train`에만 적용 (`r={0.01,0.05,0.10,0.20}`)
- 누수 방지: 정규화/FBCSP/생성학습/QC 통계는 모두 train-only
- 검증셋(`T_val`)은 early stopping/선택 전용, 테스트셋(`E_test`)은 최종 1회
- 증강 스윕은 `rho = N_synth/N_real` 기준으로 수행하고, 필요 시 `alpha_tilde=rho/(1+rho)`로 해석
- 학습 스텝 수 혼란 변수를 막기 위해 고정 스텝 학습(`step_control`)을 사용
- 시드 반복으로 평균±표준편차/신뢰구간 보고
