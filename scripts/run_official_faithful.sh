#!/usr/bin/env bash
set -euo pipefail
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export JOBLIB_TEMP_FOLDER="${JOBLIB_TEMP_FOLDER:-/tmp}"

# Usage examples:
#   ./scripts/run_official_faithful.sh
#   SEED=1 TAG=seed1 ./scripts/run_official_faithful.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_pipeline_lib.sh"
ROOT="$(pipeline_repo_root)"
cd "${ROOT}"

SEED="${SEED:-0}"
TAG="${TAG:-}"
CONFIG_PATH="${CONFIG_PATH:-configs/official_faithful.yaml}"

resolve_python

ARGS=(scripts/10_run_official_faithful_track.py --config "${CONFIG_PATH}" --seed "${SEED}")
if [[ -n "${TAG}" ]]; then
  ARGS+=(--tag "${TAG}")
fi

"${PY[@]}" "${ARGS[@]}"
