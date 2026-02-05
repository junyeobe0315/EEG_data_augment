#!/usr/bin/env bash
set -euo pipefail
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export JOBLIB_TEMP_FOLDER="${JOBLIB_TEMP_FOLDER:-/tmp}"

die() {
  echo "[error] $*" >&2
  exit 1
}

ensure_dir() {
  mkdir -p "$1"
}

# Optional: CONDA_ENV=EEG ./scripts/run_main_pipeline.sh
if [[ -n "${CONDA_ENV:-}" ]]; then
  PY=("conda" "run" "-n" "${CONDA_ENV}" "--no-capture-output" "python")
else
  if command -v python >/dev/null 2>&1; then
    PY=("python")
  elif command -v python3 >/dev/null 2>&1; then
    PY=("python3")
  else
    die "python/python3 not found. Install python or set CONDA_ENV."
  fi
fi
echo "[info] Using python: ${PY[*]}"

GEN_BATCH=""
CLF_BATCH=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gen_batch|--gen-batch)
      GEN_BATCH="$2"
      shift 2
      ;;
    --gen_batch=*|--gen-batch=*)
      GEN_BATCH="${1#*=}"
      shift 1
      ;;
    --clf_batch|--clf-batch)
      CLF_BATCH="$2"
      shift 2
      ;;
    --clf_batch=*|--clf-batch=*)
      CLF_BATCH="${1#*=}"
      shift 1
      ;;
    *)
      echo "[warn] Unknown option: $1" >&2
      shift 1
      ;;
  esac
done

GEN_ARGS=()
CLF_ARGS=()
if [[ -n "${GEN_BATCH}" ]]; then
  GEN_ARGS+=(--batch-size "${GEN_BATCH}")
fi
if [[ -n "${CLF_BATCH}" ]]; then
  CLF_ARGS+=(--batch-size "${CLF_BATCH}")
fi

ensure_dir data/raw
ensure_dir data/processed
ensure_dir data/splits
ensure_dir runs/gen
ensure_dir runs/synth
ensure_dir runs/synth_qc
ensure_dir runs/clf
ensure_dir results/metrics
ensure_dir results/tables
ensure_dir results/figures

RAW_DIR="data/raw/BCICIV_2a_gdf"
if [[ ! -d "${RAW_DIR}" && -d "./BCICIV_2a_gdf" ]]; then
  ln -s "$(pwd)/BCICIV_2a_gdf" "${RAW_DIR}"
  echo "[info] symlinked dataset -> ${RAW_DIR}"
fi
if [[ ! -d "${RAW_DIR}" ]]; then
  die "dataset not found at ${RAW_DIR}. Place BCICIV_2a_gdf under data/raw or project root."
fi

"${PY[@]}" scripts/00_prepare_data.py
[[ -s data/processed/index.csv ]] || die "processed index missing: data/processed/index.csv"
"${PY[@]}" scripts/01_make_splits.py
"${PY[@]}" - <<'PY' || die "no split files found under data/splits"
from pathlib import Path
splits = list(Path("data/splits").glob("subject_*_seed_*_p_*.json"))
if not splits:
    raise SystemExit(1)
PY
"${PY[@]}" scripts/02_train_gen.py "${GEN_ARGS[@]}"
if ! ls runs/gen/*/ckpt.pt >/dev/null 2>&1; then
  die "generator checkpoints missing under runs/gen/*/ckpt.pt"
fi
"${PY[@]}" scripts/03_sample_and_qc.py
if ! ls runs/synth/*.npz runs/synth_qc/*.npz >/dev/null 2>&1; then
  die "synthetic samples missing under runs/synth or runs/synth_qc"
fi
"${PY[@]}" scripts/04_train_clf.py "${CLF_ARGS[@]}"
if ! ls results/metrics/clf_*.csv >/dev/null 2>&1; then
  die "classifier metrics missing under results/metrics"
fi
"${PY[@]}" scripts/05_eval_and_aggregate.py
if ! ls results/tables/*.csv >/dev/null 2>&1; then
  die "aggregation tables missing under results/tables"
fi

echo "[info] Main sweep finished with evaluate_test=false by default."
echo "[info] For final test-only evaluation, run:"
echo "       ${PY[*]} scripts/05b_final_test_eval.py --input-csv results/metrics/clf_cross_session.csv --output-csv results/metrics/clf_cross_session_test.csv"
echo "       ${PY[*]} scripts/05_eval_and_aggregate.py --metrics-file clf_cross_session_test.csv"
