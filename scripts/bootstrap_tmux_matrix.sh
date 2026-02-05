#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

echo "[step1] create required directories"
mkdir -p \
  data/raw \
  data/raw/true_labels \
  data/processed \
  data/splits \
  runs/gen \
  runs/synth \
  runs/synth_qc \
  runs/clf \
  results/metrics \
  results/tables \
  results/figures

if [[ -d "${ROOT}/BCICIV_2a_gdf" && ! -e "${ROOT}/data/raw/BCICIV_2a_gdf" ]]; then
  ln -s "${ROOT}/BCICIV_2a_gdf" "${ROOT}/data/raw/BCICIV_2a_gdf"
  echo "[ok] symlinked dataset -> data/raw/BCICIV_2a_gdf"
fi

echo "[step2] install packages and python venv"
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y python3 python3-venv python3-pip tmux nvtop htop
fi

if [[ ! -d "${HOME}/.venvs/eeg" ]]; then
  python3 -m venv "${HOME}/.venvs/eeg"
fi

source "${HOME}/.venvs/eeg/bin/activate"
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt

echo "[step3] prepare true labels"
LABEL_DST="${ROOT}/data/raw/true_labels"
found=0
for src in "${ROOT}/BCICIV_2a_gdf" "${ROOT}/data/raw/BCICIV_2a_gdf"; do
  if ls "${src}"/A??E.mat >/dev/null 2>&1; then
    cp -u "${src}"/A??E.mat "${LABEL_DST}/"
    found=1
  fi
done
if [[ "${found}" -eq 0 ]]; then
  echo "[warn] AxxE.mat not found. Place true labels under ${LABEL_DST}"
fi

echo "[step4] prepare pipeline (data -> splits -> validation)"
python scripts/00_prepare_data.py
python scripts/01_make_splits.py
python scripts/11_validate_pipeline.py

echo "[step5] launch tmux sessions"
BATCHES=(256 512 1024)
for gen_bs in "${BATCHES[@]}"; do
  for clf_bs in "${BATCHES[@]}"; do
    session="genb${gen_bs}clfb${clf_bs}"
    if tmux has-session -t "${session}" 2>/dev/null; then
      echo "[skip] tmux session exists: ${session}"
      continue
    fi
    tmux new -d -s "${session}" "cd \"${ROOT}\" && source \"${HOME}/.venvs/eeg/bin/activate\" && ./scripts/run_main_pipeline.sh --gen_batch=${gen_bs} --clf_batch=${clf_bs}"
    echo "[spawned] ${session}"
  done
done

echo "[done] tmux sessions started. Check with: tmux ls"
