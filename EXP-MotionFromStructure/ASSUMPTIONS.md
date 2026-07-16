# Assumptions to confirm before scaling up

These are decisions I (the agent) made or guesses I baked into defaults.
Please confirm or correct each one before running the pipeline beyond the 3-sequence pilot.

---

## A9. Depth-convention is `ray`, not `z`, on this fisheye

**Decision (2026-04-25):** lift pixels to 3D using `P = depth × unit_ray`
(Euclidean-distance interpretation), not `P = depth × ray / ray_z`
(perspective-Z interpretation).

**Why:** Depth Anything V2 was trained on perspective images. On a 195°-FOV
fisheye, off-axis pixels in the network's view have distortion the network
doesn't model, and its scalar output is empirically closer to "distance along
the ray" than "distance along the optical axis." Single-frame test:

| convention | recovered cam height | RANSAC inliers | normal direction |
| ---------- | -------------------: | -------------: | ---------------- |
| `z`        | 12.11 m              | 3892 / 50000   | tilted, off-vertical |
| **`ray`**  | **4.90 m**           | **14859 / 50000** | nearly vertical |

Even with `ray`, absolute heights are still ~3× too high vs. the assumed
~1.5 m mount. That residual is the fundamental fisheye-on-perspective-depth-model
error and only fully goes away by undistorting to a perspective view first
(option A in the conversation log) — to be decided.

---

## A8. Architecture pivot — minimum-viable, SfM-free per-frame pipeline

**Decision (2026-04-25):** the original per-sequence SfM pipeline (stages 2–5
as designed) is overkill for the stated hypothesis. The minimum we need to
turn pixels into metric heights is:

1. Camera intrinsics — solved ONCE via stage 1 calibration on 25 frames.
2. Per-pixel metric depth — Depth-Anything-V2 (zero-shot, no fine-tuning).
3. A ground plane in the camera frame — fitted per-frame to `road` pixels.

With those, vegetation/terrain height = signed distance from each canopy point
to the fitted ground plane. **No per-sequence SfM required.** The previous
files (`stage1_run_sfm.py`, `stage3_align_depth.py`, `stage4_vegetation_height.py`)
are kept on disk but superseded — the active pipeline is now:
`stage1_calibrate.py` → `stage2_per_frame_heights.py` → `stage5_diagnostics.py`.

Mount-height recovery (originally a stage 1 SfM goal — see A1) now falls out
naturally per-frame as the camera-to-ground-plane distance, and stage 5 reports
the cross-frame median + spread.

---

## A1. Camera mount height — DISCOVERY TARGET, not an input

**Decision (2026-04-25):** the mount height is unknown and we are *not* going to
guess it. Recovering it from SfM is one of the goals — the whole point is that
GPS + altitude pin the metric scale, and the ground plane fit then tells us
where the camera sits relative to the road.

**Implication for stage 2:** drop the hard sanity-check window. Instead:
  * Report the recovered camera height per sequence.
  * Report the cross-sequence median + spread. If the rig was the same throughout
    the capture, the median across sequences is our best estimate of the mount height
    and the spread quantifies SfM noise.
  * Only flag a sequence as suspect if its recovered height is far (>3σ) from the
    cross-sequence median.

`config.yaml` → `ground_plane.expected_camera_height_m: null` (disabled),
`ground_plane.cross_sequence_outlier_sigma: 3.0`.

---

## A2. Depth model — zero-shot is fine for this experiment

**Decision (2026-04-25):** stay zero-shot with
`depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf`. We only revisit
fine-tuning if this experiment produces a strong signal that metric heights
help labelling/classification. Stage 3's per-frame scale-alignment to the SfM
ground plane is therefore the load-bearing piece — it is what makes
zero-shot metric depth usable.

---

## A3. Camera model — fisheye

**Observed in metadata:** `camera_type: "fisheye"`, dimensions 2704×2028.

**Assumption:** OpenSfM `fisheye` projection, **unknown intrinsics** (let SfM solve
them). We do *not* trust any EXIF focal length and we let OpenSfM estimate
distortion from scratch per sequence.

**Confirm:** is the same physical camera used across all sequences? If yes, we
should later pool intrinsics across sequences (more stable). If different rigs
were used, per-sequence solves are correct.

---

## A4. GPS + altitude as scale anchor

**Assumption:** the `altitude` field in metadata is **ellipsoidal/orthometric metres
above sea level** (we treat it as ENU local vertical). Combined with lon/lat, this
fixes SfM scale.

**Confirm:** is altitude reliable? Mapillary altitude can be noisy or zero. If
unreliable, we'll fall back to GPS-horizontal-only scaling (less accurate
vertically). Stage 1 logs altitude variance per sequence so we can diagnose.

---

## A5. Sequence selection thresholds

**Defaults in `config.yaml`:**
- `min_frames_per_sequence: 15`
- `max_gps_gap_m: 25` (any inter-frame jump above this splits the sequence)
- `min_track_length_m: 50` (total path length)
- `exclude_panoramas: true`

These are starting heuristics. Tune after looking at stage-0 output.

---

## A6. Segmentation = SegFormer Cityscapes; `terrain` is the grass class

**Decision (2026-04-25):** use `nvidia/segformer-b0-finetuned-cityscapes-512-1024`
(already in use elsewhere in this project).

Class mapping for our purposes:
- **`road` (id 0)** — surface used to fit the ground plane.
- **`terrain` (id 9)** — the actual GRASS target. The user confirmed: in this
  dataset, "vegetation" tends to capture trees and large bushes, while
  roadside grass is more reliably picked up by the `terrain` class.
- **`vegetation` (id 8)** — secondary class (trees / bushes). Heights are
  computed for it too, but downstream analysis should treat `terrain` as the
  measurement target for the grass-height hypothesis.

---

## A7. SfM backend is pycolmap, not OpenSfM

**Decision (2026-04-25):** stage 1 uses **pycolmap** (COLMAP 4.x Python bindings)
rather than OpenSfM.

Why we changed:
- OpenSfM has no PyPI wheel; native build requires sudo apt deps.
- OpenSfM is effectively unmaintained and known to be flaky on Python 3.12.
- Mapillary does not publish a Docker image for it.
- pycolmap pip-installs cleanly on Python 3.12, ships fisheye support, and has
  a one-call GPS alignment helper (`align_reconstruction_to_locations`) that
  fits this pipeline cleanly.

How metric scale is anchored: COLMAP itself produces a reconstruction in
arbitrary units. After incremental mapping, stage 1 builds GPS-derived ENU
coordinates per registered frame (origin = first frame's lon/lat/altitude),
then computes the similarity transform that aligns camera centres to those
coordinates and warps the whole reconstruction (cameras + points). Result is
metric — the ground plane fit in stage 2 reads xyz in metres.
