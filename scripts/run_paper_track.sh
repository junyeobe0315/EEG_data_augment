#!/usr/bin/env bash
set -euo pipefail
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export JOBLIB_TEMP_FOLDER="${JOBLIB_TEMP_FOLDER:-/tmp}"

# Usage examples:
#   ./scripts/run_paper_track.sh
#   MODELS="eegnet_tf_faithful,svm,eeg_conformer,ctnet" SEED=0 ./scripts/run_paper_track.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_pipeline_lib.sh"
ROOT="$(pipeline_repo_root)"
cd "${ROOT}"

SEED="${SEED:-0}"
SUBJECTS="${SUBJECTS:-1,2,3,4,5,6,7,8,9}"
MODELS="${MODELS:-eegnet_tf_faithful,svm,eeg_conformer,ctnet}"
EPOCH_CAP="${EPOCH_CAP:-0}"

resolve_python

"${PY[@]}" scripts/08_prepare_paper_track_data.py
"${PY[@]}" scripts/09_run_paper_track.py \
  --models "${MODELS}" \
  --subjects "${SUBJECTS}" \
  --seed "${SEED}" \
  --epoch-cap "${EPOCH_CAP}"
