#!/usr/bin/env bash

pipeline_repo_root() {
  local caller="${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}"
  (cd "$(dirname "${caller}")/.." && pwd)
}

die() {
  echo "[error] $*" >&2
  exit 1
}

ensure_dirs() {
  local d
  for d in "$@"; do
    mkdir -p "${d}"
  done
}

ensure_common_dirs() {
  ensure_dirs \
    data/raw \
    data/processed \
    data/splits \
    runs/gen \
    runs/synth \
    runs/synth_qc \
    runs/clf \
    results/metrics \
    results/tables \
    results/figures
}

ensure_dataset_link() {
  local root="$1"
  local raw_dir="${root}/data/raw/BCICIV_2a_gdf"
  if [[ ! -d "${raw_dir}" && -d "${root}/BCICIV_2a_gdf" ]]; then
    ln -s "${root}/BCICIV_2a_gdf" "${raw_dir}"
    echo "[info] symlinked dataset -> ${raw_dir}"
  fi
}

resolve_python() {
  local conda_env="${CONDA_ENV:-}"
  if [[ -n "${conda_env}" ]]; then
    PY=("conda" "run" "-n" "${conda_env}" "--no-capture-output" "python")
  else
    if command -v python >/dev/null 2>&1; then
      PY=("python")
    elif command -v python3 >/dev/null 2>&1; then
      PY=("python3")
    else
      die "python/python3 not found. Install python or set CONDA_ENV."
    fi
  fi
}

setup_runtime_env() {
  export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
  export JOBLIB_TEMP_FOLDER="${JOBLIB_TEMP_FOLDER:-/tmp}"
}
