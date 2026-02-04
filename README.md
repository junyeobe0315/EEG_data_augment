# EEG Data Augmentation for Low-Data Cross-Session MI Decoding

Reproducible pipeline to test whether **generative augmentation** improves **cross-session generalization** on **BCI Competition IV-2a** under **low-data** constraints.

## Project Scope
- Dataset: BCI Competition IV-2a (A01-A09), 4-class motor imagery.
- Main task: train on session `T`, evaluate on session `E` (cross-session).
- Main question: does synthetic EEG improve test performance when `T_train` is small?
- Core outputs: per-subject metrics, aggregated tables/figures, and distance-vs-gain analysis.

## Experiment Tracks

### 1) Main Protocol Track (recommended)
- Strict leakage-safe workflow.
- Uses saved split/subsample indices.
- Uses `rho = N_synth / N_real` with derived `alpha_tilde = rho/(1+rho)`.
- Default sweep is **validation-driven** (`evaluate_test=false`), then final test-only pass.

### 2) Paper-Faithful Track
- Isolated track for paper-style reproduction.
- Allows reference-style settings that may differ from main strict protocol.
- See `PAPER_TRACK.md`.

### 3) Official-Faithful Track
- PyTorch implementations aligned to official EEGNet/ATCNet settings.
- Used as reproducibility-defense evidence.

## Current Classifier Set
- Main sweep models: `eegnet_tf_faithful`, `svm`(FBCSP+SVM), `eeg_conformer`, `ctnet`
- Removed from main sweep: `atcnet_tf_faithful` (acceptance gap > threshold)

## Repository Layout
```text
configs/                 # YAML experiment controls
data/                    # raw, processed, splits
runs/                    # checkpoints, logs, sampled synthetic data
results/                 # metrics, tables, figures
src/                     # core modules
scripts/                 # stage scripts + run wrappers
```

## Environment
```bash
conda activate EEG
pip install -r requirements.txt
```

## Quick Start

### A) Main pipeline (00-05)
```bash
CONDA_ENV=EEG ./scripts/run_main_pipeline.sh
```

### B) Final test-only evaluation (after selecting configs on T_val)
```bash
conda run -n EEG --no-capture-output python scripts/05b_final_test_eval.py \
  --input-csv results/metrics/clf_cross_session.csv \
  --output-csv results/metrics/clf_cross_session_test.csv

conda run -n EEG --no-capture-output python scripts/05_eval_and_aggregate.py \
  --metrics-file clf_cross_session_test.csv
```

### C) Pipeline smoke check
```bash
conda run -n EEG --no-capture-output python scripts/11_validate_pipeline.py --smoke
```

## Main Protocol Summary
Detailed rules are in `PROTOCOL.md`.

- Split: `T -> train/val`, `E -> test`.
- Low-data applies to `T_train` only.
- Fit-only-on-train: normalization, FBCSP, generator, QC statistics.
- Hyperparameter/model selection on `T_val` only.
- `E_test` is final reporting only.
- Deep classifiers use fixed-step control to remove the “more data => more optimizer steps” confound.
- Generator checkpoint is selected with deterministic `T_val` proxy (`runs/gen/.../ckpt_scores.json`, `training_meta.json`).

## Output Files (Main)
- Per-run metrics: `results/metrics/clf_cross_session.csv`
- Final test metrics: `results/metrics/clf_cross_session_test.csv`
- Aggregates: `results/tables/main_table_{acc|kappa|f1_macro}.csv`
- Statistical tests: `results/tables/stats_summary.csv`
- Distance-vs-gain: `results/tables/distance_gain_correlation.csv`
- Figures: `results/figures/*.png`

## Notes
- `C2_hard_mix` is currently a legacy hard-label mix baseline (explicitly not soft-label mixup).
- `split.test_ratio` is ignored in `cross_session` mode (session `E` is full test set).

## Korean Summary (요약)
- 이 프로젝트는 BCI IV-2a에서 **교차세션 + 저데이터** 조건에서 생성 증강의 효과를 검증합니다.
- 메인 트랙은 누수 방지 규칙을 강제하며, 검증셋으로만 선택하고 테스트셋은 최종 평가에만 사용합니다.
- `run_main_pipeline.sh`로 00~05를 실행하고, 이후 `05b_final_test_eval.py`로 최종 테스트를 분리 수행합니다.
- 재현성 근거(분할 인덱스, 시드, ckpt selection 로그, 통계 요약)를 결과 파일로 남깁니다.

## References
- EEG-ATCNet: https://github.com/Altaheri/EEG-ATCNet
- EEG-Conformer: https://github.com/eeyhsong/EEG-Conformer
- CTNet: https://github.com/snailpt/CTNet
- EEGNet reference repo: https://github.com/amrzhd/EEGNet
- BCI IV-2a results: https://www.bbci.de/competition/iv/results/#dataset2a
