#!/usr/bin/env bash
set -euo pipefail
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export JOBLIB_TEMP_FOLDER="${JOBLIB_TEMP_FOLDER:-/tmp}"

# Optional: CONDA_ENV=EEG ./scripts/run_main_pipeline.sh
if [[ -n "${CONDA_ENV:-}" ]]; then
  PY=("conda" "run" "-n" "${CONDA_ENV}" "--no-capture-output" "python")
else
  PY=("python")
fi

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

"${PY[@]}" scripts/00_prepare_data.py
"${PY[@]}" scripts/01_make_splits.py
"${PY[@]}" scripts/02_train_gen.py "${GEN_ARGS[@]}"
"${PY[@]}" scripts/03_sample_and_qc.py
"${PY[@]}" scripts/04_train_clf.py "${CLF_ARGS[@]}"
"${PY[@]}" scripts/05_eval_and_aggregate.py

echo "[info] Main sweep finished with evaluate_test=false by default."
echo "[info] For final test-only evaluation, run:"
echo "       ${PY[*]} scripts/05b_final_test_eval.py --input-csv results/metrics/clf_cross_session.csv --output-csv results/metrics/clf_cross_session_test.csv"
echo "       ${PY[*]} scripts/05_eval_and_aggregate.py --metrics-file clf_cross_session_test.csv"
