# Cross-Session Low-Data Protocol (BCI IV-2a)

## 1) Goal
Validate whether generative EEG augmentation improves cross-session generalization under low-data conditions, with strict leakage control and reproducibility.

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
- Augmentation strength `alpha`: `{0.0, 0.25, 0.5, 1.0, 2.0}`
- Conditions:
  - `C0_no_aug`
  - `C1_classical` (time-shift + jitter + channel dropout)
  - `C2_mixup`
  - `GenAug_{generator}` with QC on/off
- For `GenAug`, `alpha` is synthetic:real ratio.

## 6) Distance-vs-Gain Analysis
- Embedding `phi(x)`: frozen EEGNet baseline checkpoint per split.
- Distances per class:
  - Sliced Wasserstein
  - MMD (RBF)
- Analyze relation between distance and accuracy gain (`gain_acc`) over runs.
- Outputs include scatter plot and correlation/regression tables.

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
  - `results/figures/accuracy_vs_alpha.png`
  - `results/figures/accuracy_vs_r.png`
  - `results/figures/distance_vs_gain.png`

---

# 한국어 요약

- 고정 프로토콜: `cross-session` (`T`로 학습/검증, `E`로 최종 테스트)
- low-data는 `T_train`에만 적용 (`r={0.01,0.05,0.10,0.20}`)
- 누수 방지: 정규화/FBCSP/생성학습/QC 통계는 모두 train-only
- 검증셋(`T_val`)은 early stopping/선택 전용, 테스트셋(`E_test`)은 최종 1회
- `alpha` 스윕과 baseline(`alpha=0`)을 항상 포함
- 시드 반복으로 평균±표준편차/신뢰구간 보고
