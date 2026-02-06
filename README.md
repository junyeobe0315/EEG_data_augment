# EEG Class-Conditional GenAug (BCI IV 2a)

## English

This repository provides a **leak-free, reproducible** pipeline to evaluate
class-conditional generative augmentation for EEG classification under
cross-session and low-data conditions.

Key goals:
- Verify whether GenAug improves performance under low-data regimes.
- Validate the bias–variance (approximation–estimation) trade-off in practice.
- Link performance gains to class-conditional distribution mismatch in an embedding space.

### Quick Start

1. Cache preprocessed data and build index:
```bash
python scripts/cache_preprocess.py
```

2. Prepare split indices:
```bash
python scripts/prepare_splits.py
```

3. Run a single baseline:
```bash
python scripts/run_single.py --subject 1 --seed 0 --r 0.1 --method C0 --classifier eegnet --config_pack base
```

4. Stage-1 alpha search (EEGNet only):
```bash
python scripts/run_grid.py --stage alpha_search --config_pack base
python scripts/select_alpha.py --metric val_kappa --config_pack base
```

5. Stage-2 final evaluation:
```bash
python scripts/run_grid.py --stage final_eval --config_pack base
```

6. (Optional) Hyperparameter tuning + apply tuned pack:
```bash
python scripts/tune_hparams.py --config_pack base
python scripts/apply_tuned_configs.py --best artifacts/tuning/best_params.json --out_dir configs/tuned
```

7. Generator speed comparison (runtime_sec):
```bash
python scripts/compare_generator_speed.py --results results/results.csv --classifier eegnet --out results/generator_speed.csv
```

8. Summaries / plots:
```bash
python scripts/summarize_results.py
python scripts/plot_results.py --metric kappa
```

### Directory Layout
```
configs/               # All YAML configs
scripts/               # Entry-point scripts
src/                   # Core modules
tests/                 # Leakage/shape/schema tests
artifacts/             # Splits, runs, checkpoints, figures
results/               # results.csv and summaries
```

### Reproducibility Rules (Enforced)
- Split files are the source of truth.
- All `fit()` steps use **T_train_subsample only**.
- Results are appended to one `results/results.csv`.
- Resume-safe runs: if primary key exists, the run is skipped.

### Notes
- Generators are trained only on the low-data training subset.
- Embedding φ for distance metrics is trained only on real training data.
- Distance metrics (MMD/SWD) are logged per class and aggregated by class priors.

## 한국어

이 저장소는 교차 세션 및 저데이터 조건에서 **누수 없이 재현 가능한** EEG
분류용 클래스 조건부 생성적 데이터 증강 파이프라인을 제공합니다.

핵심 목표:
- 저데이터 환경에서 GenAug 성능 향상을 검증합니다.
- 편향–분산(근사–추정) 트레이드오프를 실험적으로 확인합니다.
- 임베딩 공간에서의 클래스 조건부 분포 불일치와 성능 향상을 연결합니다.

### 빠른 시작

1. 전처리 캐시 생성 및 인덱스 구축:
```bash
python scripts/cache_preprocess.py
```

2. 스플릿 인덱스 생성:
```bash
python scripts/prepare_splits.py
```

3. 단일 베이스라인 실행:
```bash
python scripts/run_single.py --subject 1 --seed 0 --r 0.1 --method C0 --classifier eegnet --config_pack base
```

4. Stage-1 알파 서치(EEGNet만):
```bash
python scripts/run_grid.py --stage alpha_search --config_pack base
python scripts/select_alpha.py --metric val_kappa --config_pack base
```

5. Stage-2 최종 평가:
```bash
python scripts/run_grid.py --stage final_eval --config_pack base
```

6. (선택) 하이퍼파라미터 튜닝 + tuned 팩 반영:
```bash
python scripts/tune_hparams.py --config_pack base
python scripts/apply_tuned_configs.py --best artifacts/tuning/best_params.json --out_dir configs/tuned
```

7. 생성모델 속도 비교(runtime_sec):
```bash
python scripts/compare_generator_speed.py --results results/results.csv --classifier eegnet --out results/generator_speed.csv
```

8. 요약/플롯 생성:
```bash
python scripts/summarize_results.py
python scripts/plot_results.py --metric kappa
```

### 디렉터리 구조
```
configs/               # 모든 YAML 설정
scripts/               # 실행 스크립트
src/                   # 핵심 모듈
tests/                 # 누수/형상/스키마 테스트
artifacts/             # 스플릿, 런, 체크포인트, 그림
results/               # results.csv 및 요약
```

### 재현성 규칙(강제)
- 스플릿 파일이 유일한 진실입니다.
- 모든 `fit()` 단계는 **T_train_subsample만** 사용합니다.
- 결과는 하나의 `results/results.csv`에만 기록됩니다.
- 동일 PK가 있으면 실행을 건너뜁니다.

### 노트
- 생성기는 저데이터 학습 서브셋에서만 학습합니다.
- 거리 계산용 임베딩 φ는 실제 학습 데이터로만 학습합니다.
- 거리 지표(MMD/SWD)는 클래스별로 기록하고 클래스 prior로 집계합니다.
