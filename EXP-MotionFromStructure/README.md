# EXP-MotionFromStructure

Recover camera priors (intrinsics, extrinsics, metric scale) via Structure-from-Motion
on the Mapillary `santamarienseZero-shot` dataset, then combine those priors with
monocular depth estimation to produce **metric** height estimates of roadside
vegetation (cm above the road plane).

## Hypothesis
A subjective low/medium/high grass classifier is bounded by the labeller's eye.
Recovering camera intrinsics + a metric ground plane via SfM (anchored to GPS +
altitude) lets us turn depth-network outputs into auditable measurements with units.
Even if noisy, those measurements are calibrate-able. A subjective classifier is not.

## Pipeline (active stages)

After the SfM pivot (see `ASSUMPTIONS.md` A8), the active pipeline is:

| Stage | Script | Purpose |
| ----- | ------ | ------- |
| 0 | `src/stage0_select_sequences.py`    | Read `metadata.jsonl`, drop panoramas, split into sub-sequences, emit pilot manifest |
| 1 | `src/stage1_calibrate.py`           | One-shot OPENCV_FISHEYE intrinsic calibration on ~25 frames via pycolmap |
| 2 | `src/stage2_per_frame_heights.py`   | Per frame: depth + segmentation → ground plane fit → metric height map for terrain (grass) and vegetation (trees) |
| 5 | `src/stage5_diagnostics.py`         | Roll up cross-frame camera-height consistency + grass-height stability |

Superseded (kept on disk but not run): `stage1_run_sfm.py`, `stage2_ground_plane.py`,
`stage3_align_depth.py`, `stage4_vegetation_height.py`.

## Quick start

```bash
cd EXP-MotionFromStructure

# 1. read assumptions and confirm
$EDITOR ASSUMPTIONS.md

# 2. install python deps (OpenSfM is separate - see below)
pip install -r requirements.txt

# 3. stage 0 — pick pilot sequences
python -m src.stage0_select_sequences --config config.yaml

# 4. stage 1 — SfM (requires OpenSfM)
python -m src.stage1_run_sfm --config config.yaml --limit 3
```

## OpenSfM install

OpenSfM is a separate C++/Python project. The cleanest WSL install is via Docker:

```bash
docker pull opensfm/opensfm
# then in config.yaml set sfm.runner: docker
```

A native install is possible but fiddly (Ceres, OpenCV, OpenGV). See
<https://opensfm.org/docs/building.html>.

## Outputs layout

```
outputs/
├── stage0_manifest/pilot_sequences.json
├── stage1_sfm/<sequence_id>/
│   ├── opensfm_project/        # raw OpenSfM workspace
│   └── reconstruction.json     # exported metric reconstruction
├── stage2_ground/<sequence_id>/ground_plane.json
├── stage3_depth/<sequence_id>/<frame_id>.npz
├── stage4_height/<sequence_id>/<frame_id>.json
└── stage5_diag/...
```

## See also
- `ASSUMPTIONS.md` — every assumption that needs human confirmation before scaling up
- `config.yaml` — single source of truth for paths and parameters
