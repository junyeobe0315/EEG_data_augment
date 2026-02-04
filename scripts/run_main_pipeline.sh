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

"${PY[@]}" scripts/00_prepare_data.py
"${PY[@]}" scripts/01_make_splits.py
"${PY[@]}" scripts/02_train_gen.py
"${PY[@]}" scripts/03_sample_and_qc.py
"${PY[@]}" scripts/04_train_clf.py
"${PY[@]}" scripts/05_eval_and_aggregate.py
