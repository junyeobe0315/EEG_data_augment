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
    *)
      echo "[warn] Unknown option: $1" >&2
      shift 1
      ;;
  esac
done

GEN_ARGS=()
CLF_ARGS=()
QC_ARGS=()
if [[ -n "${GEN_BATCH}" ]]; then
  GEN_ARGS+=(--batch-size "${GEN_BATCH}")
fi
if [[ -n "${CLF_BATCH}" ]]; then
  CLF_ARGS+=(--batch-size "${CLF_BATCH}")
fi
if [[ "${FORCE}" -eq 1 ]]; then
  GEN_ARGS+=(--force)
  CLF_ARGS+=(--force)
  QC_ARGS+=(--force)
fi

ensure_common_dirs

RAW_DIR="data/raw/BCICIV_2a_gdf"
ensure_dataset_link "${ROOT}"
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
"${PY[@]}" scripts/03_sample_and_qc.py "${QC_ARGS[@]}"
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
