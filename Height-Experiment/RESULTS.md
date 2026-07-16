# Depth-Aware DWCGP — Results

Experiment: depth-aware modification of Verma, Zhang & Stockwell (2018)
DWCGP for ranking roadside vegetation height on Brazilian fisheye dashcam
imagery.

Run via:
```
python dwcgp_depth.py --image <full-image.jpg> --mask <verge_mask.png> \
    --output-dir Height-Experiment/results
```

## Setup

| Component | Choice |
|---|---|
| Depth model | `depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf` |
| Focal length `fy` | Estimated from HFOV = 100° → 1134.5 px (W = 2704) |
| Image set | 5 frames from `santamarienseZero-shot/raw/images/` |
| Verge masks | From `santamarienseZero-shot/raw/helpers/masks/` |
| `z_max` (depth clip) | 30 m (DA-V2 saturates near 80 m) |
| Patch clustering | DBSCAN on (X, Z) world coords, eps = 0.5 m, min_samples = 4 |

### Depth model — why not Depth Pro?

We started with `apple/DepthPro-hf` (the spec's first choice). Its HuggingFace
post-processor returned strongly compressed depth on these fisheye frames
(road bumper, sky, and far horizon all between 0.14 m and 0.33 m), which is
inconsistent with metric depth. Apple's own `canonical_inverse_depth`
formula and the HF re-implementation diverge in the (W/f) vs (f/W) factor;
either way, the post-processed values were not usable as meters here.

**Depth Anything V2 Metric Outdoor** returned plausible meters out of the
box on the same frames (hood ≈ 7 m, road mid ≈ 10 m, horizon ≈ 80 m, sky
clamped ≈ 47 m), so we used it as the metric source. It does not predict
its own focal length, so we estimate `fy` from an assumed 100° HFOV
(matching DepthPro's predicted FOV on the same images, also a typical
Brazilian dashcam). The CLI exposes `--hfov-deg` and `--focal-length` for
overrides.

### Gabor verticality — relaxation

The paper defines a pixel as "vertical" when the dominant Gabor orientation
(argmax across {0°, 45°, 90°, 135°}) is the vertical one. On these
2704×2024 fisheye dashcam frames, that strict criterion catches almost
nothing (~0.5 % of pixels) — grass blades are low-contrast at distance and
horizontal edges (blade tops, road/grass boundaries, layered leaf
texture) usually win the argmax.

We relaxed the criterion to `vertical_response > horizontal_response`
(i.e., the pixel has more vertical-stem energy than horizontal-edge
energy). This still rejects road, sky, foliage canopies, and lane lines,
and produces 30–50 % vertical-px coverage inside the grass mask — large
enough for the per-column run-length integration to be informative.

We also downscale the gray image to 1024 px wide before the filter bank
(the paper's wavelengths are tuned for ~640 px imagery) and upscale the
boolean mask back.

## Per-image summary

`fy = 1134.5 px` for all images (W = 2704, HFOV 100°).

| Image | verge px | grass / verge | vertical (in verge) | cols with l_j > 0 | min / med / max l_j (m) | # DBSCAN patches | top patch (X, Z, h_p95) |
|---|---:|---:|---:|---:|---|---:|---|
| 1054585573520650 | 845,841 | 86.8 % | 405,128 | 1,688 / 2,704 | 0.077 / **0.65** / 3.49 | 60 | (+12.6, +12.7, **2.08**) |
| 1078501224404498 | 798,625 | 81.2 % | 365,202 | 1,482 / 2,704 | 0.026 / **0.62** / 1.94 | 92 | (+14.8, +14.8, **1.84**) |
| 1132372235507698 | 644,088 | 70.3 % | 288,157 | 1,576 / 2,704 | 0.025 / **0.38** / 2.25 | 65 | (+17.3, +22.6, **2.25**) |
| 1142144651168288 | 713,089 | 83.3 % | 330,197 | 1,531 / 2,704 | 0.052 / **0.66** / 3.48 | 77 | (+14.4, +18.0, **3.48**) |
| 1185195967017805 | 648,429 | 88.9 % | 290,807 | 1,519 / 2,704 | 0.024 / **0.57** / 3.54 | 63 | (+19.7, +26.2, **2.10**) |

Median per-column run-length is **0.4–0.7 m**, which matches plausible
unmown highway-verge grass. Maxima at 3–3.5 m correspond to bushes and
small trees that fall inside the verge mask (mask is "verge strip", not
"grass-only"), and the algorithm correctly identifies them as the tallest
vertical structures.

Per-image figures (`figure.png` in each `results/<id>/` folder) show
6 panels: original + verge overlay, depth map (clipped at `z_max`), grass
mask, grass ∧ vertical mask, per-column metric height bars overlaid on
the RGB, and a top-down (X, Z) scatter of column anchors coloured by
height.

## Diagnosis

**Ranking is plausible.** Within each frame the tallest detected patches
visually correspond to the tallest vegetation (long unmown clumps and
roadside bushes), and the shortest detected patches correspond to
short/mowed sections. The top-down scatter shows two clear strips at
X ≈ ±15–25 m (left/right verges), with smooth height variation along Z.

**Caveats — three failure modes the prints surface, none of them fatal:**

1. **Lateral X is over-scaled.** Grass right beside a ~7 m road is being
   placed at X ≈ ±20 m. This is the focal-length assumption combined with
   fisheye distortion: our `fy = 1134 px` (HFOV 100°) is probably too
   small (i.e., real HFOV is wider) AND Depth Anything V2 was trained on
   rectilinear imagery, so its "metric" depth doesn't map cleanly through
   any single pinhole `fy`. Lateral positions are useful for *clustering*
   neighbouring columns into patches but not for absolute X readout. Z
   (forward distance) is more trustworthy.
2. **Verge mask includes non-grass tall verticals.** Bushes, fence
   wires, and posts inside the mask end up as the top-ranked patches
   (heights 2–3.5 m). For a height-ranking-of-grass system we'd want a
   stricter mask or an explicit "tall non-grass" reject step. For now the
   algorithm is doing the right thing — it ranks them as tall — but they
   contaminate the top of the leaderboard.
3. **Far field saturates.** Without `--z-max 30`, the longest runs
   bunched at Z ≈ 70–80 m where Depth Anything V2 plateaus, producing
   spurious tall patches (Z/fy gets large, so a few qualifying pixels
   integrate to "0.4 m of grass at 80 m"). Clipping at 30 m reproduces
   the paper's near-field assumption and the rankings stabilise.

## Recommendation

**Proceed.** The depth-aware modification produces a ranked, spatially
anchored list of vegetation patches with metric heights in a defensible
range (median 0.4–0.7 m, top quartile 1–2 m). This is sufficient to be
useful as a within-image height-ranking signal. The next-step targets,
in priority order:

1. **Tighten the verge mask to grass-only** — either an extra HSV/colour
   step or a small grass classifier. Without this, ranking will keep
   surfacing fences and bushes as the tallest "patches".
2. **Calibrate `fy`** — once we have one frame with a known reference
   (e.g., a guardrail post of known 0.75 m height), a single
   multiplicative correction on `Z/fy` will fix absolute scale.
3. **Cross-image normalisation** — heights are currently directly
   comparable only *within* a frame. Camera height variation across
   videos and depth-model bias will shift the absolute scale per image.

Out of scope for this experiment, all explicitly: the ANN grass
classifier, biomass calibration, temporal aggregation, fisheye
undistortion, depth-model A/B testing.

## Files

```
Height-Experiment/
├── dwcgp_depth.py              # the script (~330 lines)
├── RESULTS.md                  # this file
└── results/<image_id>/
    ├── figure.png              # 6-panel visualisation
    ├── report.txt              # numerical sanity stats + ranked patch table
    └── columns.npz             # per-column anchors + heights (for any later analysis)
```
