# Roadside Vegetation Classification Pipeline — Reference Specification

**Task:** Ordinal classification of roadside vegetation severity from monocular dashcam frames.
**Output:** Class ∈ {baixa, média, alta} + calibrated confidence.
**Regime:** 300–1000 labeled samples.

---

## Pipeline Overview

```
Dashcam Frame (RGB)
    │
    ▼
[1] SegFormer → 19-class semantic map
    │
    ▼
[2] ROI Extraction → left_mask, right_mask + bounding boxes
    │
    ├──────────────┬──────────────┐
    ▼              ▼              ▼
[3a] Explicit    [3b] DINOv2    [3c] Context
   Features        Embeddings     Features
  (per side)      (per side)      (global)
    │              │              │
    └──────┬───────┴──────┬───────┘
           ▼              ▼
    [4] Fusion MLP (intermediate concatenation)
           │
           ▼
    [5] CORN ordinal head (K−1 logits)
           │
           ▼
    [6] Per-threshold temperature scaling
           │
           ▼
    [7] Decision policy → class + confidence + routing
```

---

## Stage 1 — Semantic Segmentation

| Field | Value |
|---|---|
| Model | `nvidia/segformer-b2-finetuned-cityscapes-1024-1024` |
| Status | Frozen |
| Input | RGB, resized to 1024×1024 |
| Output | Per-pixel class map, 19 Cityscapes classes |
| Classes used | 8 (vegetation), 9 (terrain) — merged into `veg_mask` |
| Classes retained for context | road, sky, fence, building, pole |
| Fallback | Fine-tune on 100 hand-labeled masks if zero-shot error > 15% |

---

## Stage 2 — ROI Extraction

### Primary method: static trapezoids (v1)

| ROI | Polygon (x, y) at 640×480 | Rationale |
|---|---|---|
| `left_roi` | (0, 480), (0, 260), (280, 260), (180, 480) | Left-shoulder wedge toward vanishing point |
| `right_roi` | (640, 480), (640, 260), (360, 260), (460, 480) | Mirror of left |

### Fallback method: dynamic ROI (v2, if curved roads fail)

1. Extract road polygon from SegFormer (class 0).
2. Trace left and right road edges as polylines.
3. Offset outward by N pixels of shoulder width (N = 80 default).

### Output per image
- `left_mask`: `veg_mask ∩ left_roi`
- `right_mask`: `veg_mask ∩ right_roi`
- `left_crop`: tight bounding box of `left_mask`, non-mask pixels zeroed, padded to square
- `right_crop`: same for right side

**Design decision:** `left_crop` and `right_crop` are **tight-bbox + blackened-outside-mask**. Rationale: raw frames contain high-magnitude distractors (sky ~40%, adjacent field, cabin framing) that DINOv2's frozen attention cannot learn to suppress. Zeroed regions outside the mask produce low-variance patch tokens with negligible CLS contribution. Bounding-box dimensions are recorded as explicit features to preserve spatial context lost in cropping.

---

## Stage 3a — Explicit Features (per side: left, right)

### Photometric (7 features)

| Feature | Definition |
|---|---|
| `hsv_h_mean`, `hsv_h_std` | Hue mean and std inside mask |
| `hsv_s_mean`, `hsv_s_std` | Saturation mean and std |
| `hsv_v_mean` | Value mean |
| `exg_mean`, `exg_std` | Excess Green = 2G − R − B, mean and std |
| `green_chroma` | G / (R + G + B), mean |
| `lab_b_mean` | L*a*b* b-channel mean (yellow-blue axis) |

### Geometric (6 features)

| Feature | Definition |
|---|---|
| `mask_px_count` | Vegetation area in pixels |
| `mask_density` | Mask area / ROI trapezoid area |
| `vertical_extent` | max_y − min_y of mask pixels |
| `centroid_y_norm` | Centroid y-coord / image height |
| `bbox_aspect_ratio` | bbox width / bbox height |
| `top_vs_horizon` | topmost mask point y / horizon y |

### Texture (3 features)

| Feature | Definition |
|---|---|
| `glcm_contrast` | GLCM contrast on masked grayscale |
| `glcm_homogeneity` | GLCM homogeneity |
| `edge_density` | Canny edge fraction inside mask |

### Depth (optional, 3 features — requires Depth Anything V2)

| Feature | Definition |
|---|---|
| `depth_mean_in_mask` | Mean predicted depth inside mask |
| `depth_std_in_mask` | Std of predicted depth |
| `height_proxy` | vertical_extent × mean_depth_gradient_at_mask_base |

### Bounding box context (4 features — added to compensate for crop)

| Feature | Definition |
|---|---|
| `bbox_w_norm`, `bbox_h_norm` | bbox width/height / image dims |
| `bbox_area_norm` | bbox area / image area |
| `bbox_centroid_x_norm` | bbox centroid x / image width |

**Total per side:** ~23 features (20 without depth). **Total both sides:** ~46 features.

---

## Stage 3b — DINOv2 Embeddings

| Field | Value |
|---|---|
| Model | `facebook/dinov2-base` |
| Status | Frozen (inference only) |
| Input | `left_crop` and `right_crop` (224×224, RGB, ImageNet normalized) |
| Output | CLS token, 768-dim per crop |
| Caching | Compute once, save as `.npy` per image |

**Total:** 768 × 2 = 1536 features.

---

## Stage 3c — Global Context Features

Computed once per image from the full SegFormer output.

| Feature | Definition |
|---|---|
| `road_frac` | Fraction of pixels classified as road |
| `sky_frac` | Fraction classified as sky |
| `veg_frac_total` | Fraction classified as vegetation + terrain |
| `fence_frac`, `building_frac`, `pole_frac` | Respective class fractions |
| `horizon_y_norm` | Estimated horizon y / image height |

**Total:** ~7 features.

---

## Stage 4 — Fusion Architecture

### Branch structure

```
┌─────────────────────────────────────────────────────────┐
│ TABULAR BRANCH                                          │
│                                                         │
│  left_explicit  (23)  ┐                                 │
│  right_explicit (23)  ├─→ concat (53) → Linear(64)      │
│  global_context (7)   ┘      │                          │
│                              ▼                          │
│                           BN + ReLU + Dropout(0.4)      │
│                              │                          │
│                              ▼                          │
│                          Linear(32) → BN+ReLU           │
│                              │                          │
│                              ▼ (32-dim)                 │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ IMAGE BRANCH                                            │
│                                                         │
│  left_dino  (768)  ┐                                    │
│                    ├─→ concat (1536) → Linear(512)      │
│  right_dino (768)  ┘         │                          │
│                              ▼                          │
│                           BN + ReLU + Dropout(0.3)      │
│                              │                          │
│                              ▼                          │
│                          Linear(256) → BN+ReLU          │
│                              │                          │
│                              ▼ (256-dim)                │
└─────────────────────────────────────────────────────────┘

    tabular(32) ⊕ image(256) = concat(288)
                  │
                  ▼
             Linear(128) → BN+ReLU+Dropout(0.3)
                  │
                  ▼
          Linear(K−1 = 2)  ← CORN output head
```

### Regularization

| Component | Value |
|---|---|
| Dropout (tabular) | 0.4 |
| Dropout (image) | 0.3 |
| Dropout (fusion) | 0.3 |
| Weight decay | 1e-4 |
| BatchNorm | After every Linear before ReLU |

---

## Stage 5 — Loss Function (CORN)

### Mathematical formulation

For K=3 ordered classes, produce K−1=2 logits representing conditional probabilities:
- `logit_0` → P(y ≥ média)
- `logit_1` → P(y ≥ alta | y ≥ média)

### Class probability recovery at inference

```
p(baixa) = 1 − σ(logit_0)
p(média) = σ(logit_0) · (1 − σ(logit_1))
p(alta)  = σ(logit_0) · σ(logit_1)
```

### Loss computation

Conditional BCE with per-threshold positive weights:
- Threshold 0: BCE over all samples, target = [y ≥ 1]
- Threshold 1: BCE over samples where y ≥ 1, target = [y ≥ 2]

### Implementation

```python
from coral_pytorch.losses import corn_loss
loss = corn_loss(logits, labels, num_classes=3)
```

### Guarantees
- Rank monotonicity by construction (chain rule).
- Full per-threshold model capacity (no shared-weight constraint).
- Graceful degradation: under-confidence biases toward middle class.

---

## Stage 6 — Calibration

### Method: per-threshold temperature scaling

Two scalars `T_0`, `T_1` fit on held-out validation via NLL minimization on the sigmoid outputs.

```
calibrated_prob_t = σ(logit_t / T_t)
```

### Validation metrics

| Metric | Target |
|---|---|
| Expected Calibration Error (ECE) per threshold | < 0.05 |
| Reliability diagram | Monotonic, close to diagonal |

---

## Stage 7 — Decision Policy

### Confidence buckets

| Condition | Route | Action |
|---|---|---|
| `max(p) ≥ 0.80` | Automated | Emit class + confidence |
| `0.50 ≤ max(p) < 0.80` | Flagged | Emit class, mark for review |
| `max(p) < 0.50` OR top-2 gap < 0.15 | Human | Route to inspector |

### Explainability payload per prediction

| Field | Source |
|---|---|
| Predicted class + confidence | Calibrated CORN output |
| Top-3 contributing explicit features | SHAP values on tabular branch |
| 5 nearest training neighbors | Cosine similarity in DINOv2 concat space |
| Visual overlay | `veg_mask` + ROI polygons rendered on original frame |
| Per-class probability vector | Full (p_baixa, p_média, p_alta) |

---

## Training Protocol

| Setting | Value |
|---|---|
| Split | Stratified 5-fold CV, seed 42 |
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR |
| Batch size | 32 |
| Max epochs | 100 |
| Early stopping | Val macro-F1, patience 15 |
| Augmentation (on crops pre-DINOv2) | HorizontalFlip, ColorJitter(0.2), Mixup(α=0.2) |
| Class balancing | Per-threshold `pos_weight` in BCE |

---

## Evaluation Metrics

| Metric | Why report |
|---|---|
| Macro-F1 | Class-balanced performance, comparable to prior work |
| MAE (in class units: 0, 1, 2) | Primary ordinal metric — shows CORN benefit |
| Quadratic Weighted Kappa | Penalizes distant errors more than near ones |
| Per-class Precision / Recall | Operational risk analysis |
| Confusion matrix | Error structure inspection |
| ECE per threshold | Calibration quality |
| Reliability diagram | Calibration visualization |

---

## Feature Count Summary

| Source | Count |
|---|---|
| Explicit (left) | 23 |
| Explicit (right) | 23 |
| Global context | 7 |
| DINOv2 (left) | 768 |
| DINOv2 (right) | 768 |
| **Total input features** | **1589** |

---

## Dependencies

| Package | Purpose |
|---|---|
| `transformers` | SegFormer, DINOv2 inference |
| `torch`, `torchvision` | Core model |
| `coral-pytorch` | CORN loss and utilities |
| `timm` | Optional alternative backbones |
| `opencv-python`, `scikit-image` | Explicit feature computation |
| `depth-anything-v2` (optional) | Monocular depth features |
| `shap` | Explanation values |
| `scikit-learn` | Calibration, metrics, CV splits |

---

## Open Design Decisions

| Decision | Current choice | Alternative to test |
|---|---|---|
| DINOv2 input | Tight-bbox + blackened-outside-mask | Raw crop; concat of both |
| DINOv2 size | base (768-dim) | small (384-dim) for speed; large (1024-dim) for capacity |
| ROI extraction | Static trapezoids | Dynamic road-edge offsetting |
| Depth features | Optional | Always on if compute budget allows |
| Ordinal loss | CORN | CORAL (with rank-monotonicity by shared weights) |
| Calibration | Per-threshold T | Single T on softmaxed class probs |