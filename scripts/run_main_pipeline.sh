#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_pipeline_lib.sh"
ROOT="$(pipeline_repo_root)"
cd "${ROOT}"
setup_runtime_env

# Optional: CONDA_ENV=EEG ./scripts/run_main_pipeline.sh
resolve_python
echo "[info] Using python: ${PY[*]}"

GEN_BATCH=""
CLF_BATCH=""
FORCE=0
FAST=0
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
    --force)
      FORCE=1
      shift 1
      ;;
    --fast)
      FAST=1
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
QC_ARGS=()
PIPE_ARGS=()
if [[ -n "${GEN_BATCH}" ]]; then
  GEN_ARGS+=(--batch-size "${GEN_BATCH}")
  PIPE_ARGS+=(--gen-batch "${GEN_BATCH}")
fi
if [[ -n "${CLF_BATCH}" ]]; then
  CLF_ARGS+=(--batch-size "${CLF_BATCH}")
  PIPE_ARGS+=(--clf-batch "${CLF_BATCH}")
fi
if [[ "${FORCE}" -eq 1 ]]; then
  GEN_ARGS+=(--force)
  CLF_ARGS+=(--force)
  QC_ARGS+=(--force)
  PIPE_ARGS+=(--force)
fi
if [[ "${FAST}" -eq 1 ]]; then
  GEN_ARGS+=(--fast)
  CLF_ARGS+=(--fast)
  PIPE_ARGS+=(--fast)
fi

ensure_common_dirs

RAW_DIR="data/raw/BCICIV_2a_gdf"
ensure_dataset_link "${ROOT}"
if [[ ! -d "${RAW_DIR}" ]]; then
  die "dataset not found at ${RAW_DIR}. Place BCICIV_2a_gdf under data/raw or project root."
fi

"${PY[@]}" main.py pipeline "${PIPE_ARGS[@]}"

echo "[info] Main sweep finished with evaluate_test=false by default."
echo "[info] For final test-only evaluation, run:"
echo "       ${PY[*]} main.py final-test --input-csv results/metrics/clf_cross_session.csv --output-csv results/metrics/clf_cross_session_test.csv"
echo "       ${PY[*]} main.py eval-aggregate --metrics-file clf_cross_session_test.csv"
