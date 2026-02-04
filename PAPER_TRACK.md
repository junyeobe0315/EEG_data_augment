# Paper-Faithful Track Guide

This track is intentionally separated from the main protocol.
Use it for **paper/repo comparability**, not for final leakage-safe augmentation claims.

## Why This Exists
- Defend implementation fidelity when releasing code.
- Reproduce public settings as closely as possible.
- Keep differences from the main protocol explicit and auditable.

## Entry Points
- `scripts/08_prepare_paper_track_data.py`
- `scripts/09_run_paper_track.py`
- `scripts/10_run_official_faithful_track.py`
- `configs/paper_track.yaml`
- `configs/official_faithful.yaml`

## Data Variants
- `atcnet_style`: trial-start window `1.5-6.0s` (1125 samples)
- `eeg_conformer_style`: trial-start window `2.0-6.0s` (1000 samples), Cheby2 `4-40Hz`
- `ctnet_style`: cue-onset window `0.0-4.0s` (1000 samples)

Generated files:
- `data/paper_track/<track>/Sxx_T.npz`
- `data/paper_track/<track>/Sxx_E.npz`
- `data/paper_track/<track>/index.csv`

## Typical Commands

### 1) Build paper-track data
```bash
conda run -n EEG --no-capture-output python scripts/08_prepare_paper_track_data.py
```

### 2) Run paper track
```bash
conda run -n EEG --no-capture-output python scripts/09_run_paper_track.py \
  --models eegnet,svm,eeg_conformer,ctnet \
  --subjects 1,2,3,4,5,6,7,8,9 \
  --seed 0
```

### 3) Official-faithful acceptance run
```bash
conda run -n EEG --no-capture-output python scripts/10_run_official_faithful_track.py --seed 0
```

### 4) Fast debug
```bash
conda run -n EEG --no-capture-output python scripts/09_run_paper_track.py \
  --models eegnet,svm,eeg_conformer,ctnet \
  --subjects 1 \
  --seed 0 \
  --epoch-cap 2
```

## Outputs
- Per-run: `results/metrics/paper_track_seed{seed}.csv`
- Summary: `results/tables/paper_track_seed{seed}_summary.csv`
- Paper comparison: `results/tables/paper_track_seed{seed}_compare.csv`
- Official-faithful comparison: `results/tables/official_faithful_seed{seed}_compare.csv`

## Important Differences vs Main Track
- May use paper-specific validation behavior (`val=test` for some settings).
- Main fixed-step confound control is disabled to preserve paper epoch behavior.
- Intended for reproducibility defense and sanity checks only.
- Final augmentation conclusions must come from `PROTOCOL.md` main track.

## Recommendation
- Keep both tracks in reports:
  - `Main Protocol` for scientific claims
  - `Paper-Faithful` for implementation-fidelity evidence
