# EEG Data Augmentation for Cross-Session Low-Data MI (BCI IV-2a)

Reproducible research pipeline for testing whether **generative EEG augmentation** improves **cross-session generalization** under **low-data** constraints.

## 1) What This Project Answers
- Dataset: BCI Competition IV-2a (`A01`-`A09`, 4-class MI)
- Train domain: session `T`
- Target domain: session `E`
- Main question: does augmentation help when only a small fraction of `T_train` is available?

## 2) Main Tracks
- `Main Protocol Track` (recommended for claims)
  - leakage-safe split/normalization/training/evaluation flow
  - validation-driven model selection (`T_val` only)
  - final test is a separate step
- `Paper-Faithful Track`
  - paper/repo-style reproduction-oriented settings
- `Official-Faithful Track`
  - PyTorch implementations aligned to official code behavior for defense of implementation fidelity

See `PROTOCOL.md` and `PAPER_TRACK.md` for details.

## 3) Models
- Classifiers (main sweep):
  - `eegnet_tf_faithful`
  - `svm` (FBCSP + SVM)
  - `eeg_conformer`
  - `ctnet`
- Generators:
  - `eeggan_net`
  - `cwgan_gp`
  - `conditional_ddpm`
  - `cvae`
- Controls:
  - `C0_no_aug`
  - `C1_classical`
  - `C2_hard_mix` (legacy hard-label mix baseline)

## 4) Reproducibility / Fairness Guarantees
- Saved and reused split/subsample indices (`data/splits/*.json`)
- Train-only fitting for normalization/FBCSP/generator/QC statistics
- Classifier best checkpoint metric is configurable (default: `kappa`)
- Generator checkpoint selected by deterministic `T_val` proxy (`ckpt_scores.json`)
- Condition-stable seeds for classifier runs and stable seeds for synth/QC sampling
- Fixed-step training option for deep classifiers to control step-count confound
- Logged requested vs effective mixture:
  - `ratio`, `alpha_tilde`
  - `ratio_effective`, `alpha_effective`

## 5) Repository Layout
```text
configs/    # YAML configs
scripts/    # CLI helpers (imported by main.py)
src/        # core modules
runs/       # checkpoints/logs/synthetic artifacts (gitignored)
results/    # metrics/tables/figures (gitignored)
data/       # raw/processed/splits (gitignored)
```

## 6) Environment Setup
```bash
conda activate EEG
pip install -r requirements.txt
```

## 7) Minimal Command Flow
Note: CLI entrypoint is unified in `main.py`. Direct execution of `scripts/*.py` is disabled.

### A. Main pipeline
```bash
CONDA_ENV=EEG ./scripts/run_main_pipeline.sh

# Speed-optimized (AMP/TF32/pin_memory) on GPU
CONDA_ENV=EEG ./scripts/run_main_pipeline.sh --fast

# Parallel run (2 jobs) across splits/models
CONDA_ENV=EEG ./scripts/run_main_pipeline.sh --jobs 2 --devices 0,1

# Runtime config overrides
CONDA_ENV=EEG ./scripts/run_main_pipeline.sh --set gen.train.batch_size=64 --set clf.train.lr=0.0005
  # Format: <config>.<path>=<value>, where config is one of: data/gen/clf/qc/sweep/split/preprocess

# Run a subset of steps
CONDA_ENV=EEG ./scripts/run_main_pipeline.sh --steps train-gen,sample-qc,train-clf

# Skip a step
CONDA_ENV=EEG ./scripts/run_main_pipeline.sh --skip eval-aggregate

# Print resolved steps only
CONDA_ENV=EEG ./scripts/run_main_pipeline.sh --print-steps
```

### B. Final test-only evaluation (after selecting configs on `T_val`)
```bash
conda run -n EEG --no-capture-output python main.py final-test \
  --input-csv results/metrics/clf_cross_session.csv \
  --output-csv results/metrics/clf_cross_session_test.csv

conda run -n EEG --no-capture-output python main.py eval-aggregate \
  --metrics-file clf_cross_session_test.csv
```

### C. Sanity checks
```bash
# Config/split integrity check
conda run -n EEG --no-capture-output python main.py validate

# Tiny smoke train
conda run -n EEG --no-capture-output python main.py validate --smoke

# One-shot quick pilot (gen + qc + baseline/genaug)
conda run -n EEG --no-capture-output python main.py pilot \
  --subject 1 --seed 0 --p 0.20 \
  --gen-model cvae --clf-model eegnet_tf_faithful \
  --ratio 0.5 --qc-on --gen-epochs 3 --clf-steps 200
```

### D. Paper / Official-faithful tracks
```bash
CONDA_ENV=EEG ./scripts/run_paper_track.sh
CONDA_ENV=EEG ./scripts/run_official_faithful.sh
```

## 8) Useful `.sh` Entry Scripts
- `scripts/run_main_pipeline.sh`
  - runs `main.py pipeline`
- `scripts/run_one.sh`
  - alias to `run_main_pipeline.sh`
- `scripts/run_paper_track.sh`
  - paper track data prep + run
- `scripts/run_official_faithful.sh`
  - official-faithful acceptance run

## 9) Core Outputs
- Run-level metrics: `results/metrics/clf_cross_session.csv`
- Final test metrics: `results/metrics/clf_cross_session_test.csv`
- Aggregated tables:
  - `results/tables/main_table_acc.csv`
  - `results/tables/main_table_kappa.csv`
  - `results/tables/main_table_f1_macro.csv`
  - `results/tables/stats_summary.csv`
  - `results/tables/distance_gain_correlation.csv`
- Figures:
  - `results/figures/accuracy_vs_ratio.png`
  - `results/figures/accuracy_vs_r.png`
  - `results/figures/distance_vs_gain.png`

## 10) Notes
- In `cross_session`, `split.test_ratio` is intentionally ignored (`E` is full test set).
- Main track default is `evaluate_test=false`; test is for final reporting only.

---

## 한국어 요약

### 프로젝트 목표
- BCI IV-2a에서 `cross-session(T->E)` + `low-data` 조건에서
- 생성모델 기반 증강이 일반화 성능을 올리는지
- 누수 없이 재현 가능하게 검증합니다.

### 핵심 원칙
- split/subsample 인덱스 저장 및 재사용
- 정규화/FBCSP/생성기/QC 통계는 train-only fit
- 모델 선택은 `T_val`만 사용
- `E_test`는 최종 평가 단계에서만 사용

### 지금 바로 실행할 명령
```bash
CONDA_ENV=EEG ./scripts/run_main_pipeline.sh

conda run -n EEG --no-capture-output python main.py final-test \
  --input-csv results/metrics/clf_cross_session.csv \
  --output-csv results/metrics/clf_cross_session_test.csv

conda run -n EEG --no-capture-output python main.py eval-aggregate \
  --metrics-file clf_cross_session_test.csv
```

### 보조 실행 스크립트
- `scripts/run_paper_track.sh`: 논문 재현용 트랙
- `scripts/run_official_faithful.sh`: 공식 코드 정합성 점검 트랙
- `scripts/run_one.sh`: 메인 파이프라인 실행

## References
- EEG-Conformer: https://github.com/eeyhsong/EEG-Conformer
- EEGNet reference: https://github.com/amrzhd/EEGNet
- EEG-ATCNet: https://github.com/Altaheri/EEG-ATCNet
- CTNet: https://github.com/snailpt/CTNet
- BCI IV results page: https://www.bbci.de/competition/iv/results/#dataset2a
