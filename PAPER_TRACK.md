# Paper-Faithful Track

This track is isolated from the main low-data augmentation pipeline and is meant for reproducibility defense when releasing code.

## Purpose
- Reproduce public-paper settings as closely as possible.
- Keep model architecture/protocol differences explicit.
- Separate these runs from the main project protocol.

## Added Config/Entry Points
- `configs/paper_track.yaml`
- `configs/official_faithful.yaml`
- `scripts/08_prepare_paper_track_data.py`
- `scripts/09_run_paper_track.py`
- `scripts/10_run_official_faithful_track.py`

## Data Tracks
- `atcnet_style`: trial-start window 1.5–6.0 s (1125 samples)
- `eeg_conformer_style`: trial-start window 2.0–6.0 s (1000 samples), Cheby2 4–40 Hz
- `ctnet_style`: cue-onset window 0.0–4.0 s (1000 samples)

Generated files:
- `data/paper_track/<track>/Sxx_T.npz`
- `data/paper_track/<track>/Sxx_E.npz`
- `data/paper_track/<track>/index.csv`

## Run
1. Prepare paper-track datasets
```bash
conda run -n EEG --no-capture-output python scripts/08_prepare_paper_track_data.py
```

2. Run paper track (example: EEGNet, all subjects)
```bash
conda run -n EEG --no-capture-output python scripts/09_run_paper_track.py \
  --models eegnet \
  --subjects 1,2,3,4,5,6,7,8,9 \
  --seed 0
```

3. Fast debug run with epoch cap
```bash
conda run -n EEG --no-capture-output python scripts/09_run_paper_track.py \
  --models eegnet,atcnet,eeg_conformer,ctnet,svm \
  --subjects 1 \
  --seed 0 \
  --epoch-cap 2
```

4. Shell wrappers
```bash
CONDA_ENV=EEG ./scripts/run_paper_track.sh
CONDA_ENV=EEG ./scripts/run_official_faithful.sh
```

## Outputs
- Per-run metrics:
  - `results/metrics/paper_track_seed{seed}.csv`
- Aggregated summary:
  - `results/tables/paper_track_seed{seed}_summary.csv`
- Paper reference comparison:
  - `results/tables/paper_track_seed{seed}_compare.csv`

## Note on Validation Behavior
- This track intentionally follows each paper/repo style, not the main strict protocol.
- For models configured with `val_ratio=0` in `configs/paper_track.yaml` (e.g., some Conformer-style settings), `scripts/09_run_paper_track.py` uses `val=test` to emulate reference behavior.
- Main-track fixed-step control (`train.step_control`) is explicitly disabled here so paper epoch settings are preserved.
- Use this only for paper-faithful comparison, not for the main leakage-safe experiment.
