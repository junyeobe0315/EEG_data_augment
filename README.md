# EEG Data Augmentation For Low-Data Cross-Session MI Decoding  
# 저데이터 교차세션 MI 디코딩을 위한 EEG 데이터 증강 연구

## English
### 1) What This Repository Does
This project studies a single core question:

**Can generative augmentation improve cross-session generalization under low-data conditions on BCI Competition IV-2a?**

The repository contains:
1. A full experiment pipeline (`00` to `05`) for preprocessing, splitting, generator training, classifier training, evaluation, and aggregation.
2. A paper-comparison track (`08`, `09`) for no-augmentation reference checks.
3. An official-faithful PyTorch track (`10`) that mirrors official EEGNet/ATCNet TensorFlow settings for reproducibility defense.
4. A protocol document with leakage rules and low-data definition: `PROTOCOL.md`.

### 2) Current Model Decision (Before Main Experiments)
Acceptance rule: **mean accuracy gap vs paper must be <= 2.0%p**.

Latest result file: `results/tables/official_faithful_seed0_compare.csv`

| Model | Ours (%) | Paper (%) | Abs Gap | Decision |
|---|---:|---:|---:|---|
| `eegnet_tf_faithful` | 68.287 | 68.670 | 0.383 | Keep |
| `atcnet_tf_faithful` | 69.252 | 81.100 | 11.848 | Drop |

Decision file: `results/tables/official_faithful_seed0_decision.csv`  
Main sweep config updated to exclude ATCNet: `configs/sweep.yaml`

### 3) Models Used And Training Differences
| Model key | Status | Training style in this repo | Difference vs source paper/repo |
|---|---|---|---|
| `eegnet_tf_faithful` | Main | Subject-specific T->E, 500 epochs, batch 64, train-only normalization (`channel_timepoint`), val-loss checkpoint | Official-faithful PyTorch implementation aligned to EEG-ATCNet EEGNet settings |
| `atcnet_tf_faithful` | Removed | Same official-faithful protocol as above | Gap > 2%p vs paper target, so excluded from main experiments |
| `svm` | Main | FBCSP features + SVM, train-only fitting | Classical baseline, not deep end-to-end |
| `eeg_conformer` | Main | Paper-track / main-track configurable | Official public code uses test-driven best-epoch behavior; this repo keeps fixed evaluation protocol |
| `ctnet` | Main | Paper-track / main-track configurable with `paper_sr` option | Reimplemented in PyTorch; follows project leakage rules |

### 4) Shell Scripts (Organized)
| Script | Purpose | Example |
|---|---|---|
| `scripts/run_main_pipeline.sh` | End-to-end main pipeline (`00`~`05`) | `CONDA_ENV=EEG ./scripts/run_main_pipeline.sh` |
| `scripts/run_paper_track.sh` | Prepare + run paper comparison track (`08`,`09`) | `CONDA_ENV=EEG MODELS=eegnet_tf_faithful,svm,eeg_conformer,ctnet ./scripts/run_paper_track.sh` |
| `scripts/run_official_faithful.sh` | Run official-faithful acceptance track (`10`) | `CONDA_ENV=EEG SEED=0 ./scripts/run_official_faithful.sh` |
| `scripts/run_one.sh` | Alias of main pipeline script | `./scripts/run_one.sh` |

### 5) Project Structure
```text
configs/                      # all experiment controls
data/
  raw/                        # raw BCI2a files + true labels
  processed/                  # main protocol processed arrays
  paper_track/                # paper-style processed arrays
  splits/                     # saved split/subsample indices
runs/
  gen/ synth/ synth_qc/ clf/  # checkpoints and logs
  external/                   # external/reference artifacts moved out of root
results/
  metrics/ tables/ figures/   # per-run + aggregated outputs
src/
  dataio.py split.py preprocess.py
  models_gen.py models_clf.py models_official_faithful.py distribution.py
  train_gen.py sample_gen.py train_clf.py train_official_faithful.py
  eval.py aggregate.py utils.py
scripts/
  00_prepare_data.py
  01_make_splits.py
  02_train_gen.py
  03_sample_and_qc.py
  04_train_clf.py
  05_eval_and_aggregate.py
  08_prepare_paper_track_data.py
  09_run_paper_track.py
  10_run_official_faithful_track.py
  11_validate_pipeline.py
  run_main_pipeline.sh
  run_paper_track.sh
  run_official_faithful.sh
  run_one.sh
```

### 6) Environment
Tested with:
1. Python 3.11
2. PyTorch 2.10 (CUDA-enabled)
3. MNE, NumPy, SciPy, scikit-learn, pandas, matplotlib

Example:
```bash
conda activate EEG
pip install -r requirements.txt
```

### 7) Minimal Repro Commands
Prepare processed data:
```bash
python scripts/00_prepare_data.py
python scripts/01_make_splits.py
```

Run official-faithful acceptance track:
```bash
python scripts/10_run_official_faithful_track.py --seed 0
```

Run core no-aug paper comparison:
```bash
python scripts/08_prepare_paper_track_data.py
python scripts/09_run_paper_track.py --models eegnet_tf_faithful,svm,eeg_conformer,ctnet --seed 0
```

Run full augmentation pipeline:
```bash
python scripts/02_train_gen.py
python scripts/03_sample_and_qc.py
python scripts/04_train_clf.py
python scripts/05_eval_and_aggregate.py
```

Pipeline integrity check (recommended before long runs):
```bash
python scripts/11_validate_pipeline.py --smoke
```

### 8) Reproducibility Rules Implemented
1. Split/subsample indices are saved and reused.
2. Normalization is train-only by design.
3. Generator/QC fitting is train-only.
4. Tuning uses validation only; test is final-only.
5. Seeds are set and logged.

### 9) Notes For Publication-Grade Release
1. Include `results/tables/official_faithful_seed0_compare.csv` in supplementary material.
2. Keep `configs/official_faithful.yaml` and `scripts/10_run_official_faithful_track.py` as your reproducibility defense track.
3. Report both per-subject and aggregated metrics (`acc`, `kappa`, `macro-F1`).

---

## 한국어
### 1) 이 저장소의 목적
이 프로젝트의 핵심 질문은 다음 하나입니다.

**저데이터 조건에서 생성 기반 증강이 BCI IV-2a 교차세션 일반화를 실제로 개선하는가?**

저장소에는 다음이 포함되어 있습니다.
1. 전처리부터 집계까지 전체 파이프라인(`00`~`05`)
2. 논문 수치 비교용 트랙(`08`, `09`)
3. 공식 구현 대응 재현 방어 트랙(`10`, PyTorch, EEGNet/ATCNet)

### 2) 본 실험 전 모델 채택 기준
채택 기준: **논문 대비 평균 정확도 차이 <= 2.0%p**

최신 비교 파일: `results/tables/official_faithful_seed0_compare.csv`

| 모델 | 우리 결과(%) | 논문(%) | 차이 | 결정 |
|---|---:|---:|---:|---|
| `eegnet_tf_faithful` | 68.287 | 68.670 | 0.383 | 유지 |
| `atcnet_tf_faithful` | 69.252 | 81.100 | 11.848 | 제외 |

결정 파일: `results/tables/official_faithful_seed0_decision.csv`  
메인 스윕에서 ATCNet 제외 반영: `configs/sweep.yaml`

### 3) 사용 모델과 학습 방식 차이
| 모델 키 | 상태 | 이 프로젝트 학습 방식 | 논문/공식 코드 대비 차이 |
|---|---|---|---|
| `eegnet_tf_faithful` | 메인 사용 | subject-specific T->E, 500 epoch, batch 64, train-only 정규화(`channel_timepoint`), val-loss 기준 ckpt 선택 | EEG-ATCNet EEGNet 설정을 PyTorch로 최대한 동일하게 반영 |
| `atcnet_tf_faithful` | 메인 제외 | 위와 동일한 official-faithful 프로토콜 | 논문 대비 2%p 기준 초과로 제외 |
| `svm` | 메인 사용 | FBCSP + SVM, train-only 피팅 | 전통 ML 베이스라인 |
| `eeg_conformer` | 메인 사용 | paper/main 트랙 설정 가능 | 공개 코드는 test 기반 epoch 선택 경향, 본 프로젝트는 고정 평가 규칙 유지 |
| `ctnet` | 메인 사용 | paper/main 트랙 설정 가능, `paper_sr` 옵션 지원 | 프로젝트 누수 방지 규칙에 맞게 PyTorch 재구현 |

### 4) `.sh` 실행 스크립트 정리
| 스크립트 | 역할 | 예시 |
|---|---|---|
| `scripts/run_main_pipeline.sh` | 메인 파이프라인 전체(`00`~`05`) | `CONDA_ENV=EEG ./scripts/run_main_pipeline.sh` |
| `scripts/run_paper_track.sh` | paper 비교 트랙(`08`,`09`) | `CONDA_ENV=EEG MODELS=eegnet_tf_faithful,svm,eeg_conformer,ctnet ./scripts/run_paper_track.sh` |
| `scripts/run_official_faithful.sh` | 공식 재현 검증 트랙(`10`) | `CONDA_ENV=EEG SEED=0 ./scripts/run_official_faithful.sh` |
| `scripts/run_one.sh` | `run_main_pipeline.sh` 별칭 | `./scripts/run_one.sh` |

### 5) 실행 순서(권장)
1. 데이터 준비: `00_prepare_data.py` -> `01_make_splits.py`
2. 재현 검증: `10_run_official_faithful_track.py`
3. 논문 비교(무증강): `08_prepare_paper_track_data.py` -> `09_run_paper_track.py`
4. 본 실험(증강 포함): `02` -> `03` -> `04` -> `05`

### 6) 재현성 원칙
1. split/subsample 인덱스 저장
2. 정규화 통계는 train-only
3. 생성/QC 학습도 train-only
4. 하이퍼파라미터 선택은 val-only
5. test는 최종 평가 1회

### 7) 공개용 체크포인트
1. `configs/official_faithful.yaml`
2. `scripts/10_run_official_faithful_track.py`
3. `results/tables/official_faithful_seed0_compare.csv`
4. `results/tables/official_faithful_seed0_decision.csv`

## References
1. EEG-ATCNet: https://github.com/Altaheri/EEG-ATCNet
2. EEG-Conformer: https://github.com/eeyhsong/EEG-Conformer
3. CTNet: https://github.com/snailpt/CTNet
4. EEGNet (reference repo): https://github.com/amrzhd/EEGNet
5. BCI Competition IV-2a results page: https://www.bbci.de/competition/iv/results/#dataset2a
