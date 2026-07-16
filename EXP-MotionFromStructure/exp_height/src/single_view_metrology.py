"""
Single-view metrology grass-height estimator + correlation test against
the human Baixo/Médio/Alto labels (annotation-v2, "certo" only).

No depth model. No SfM. The only assumption is:
  * camera mount height H above the road (we use 1.5 m by default — a typical
    vehicle dashcam mount). Heights scale linearly with H, so any constant H
    preserves the *rank ordering* across classes — which is all we need to
    test whether the geometry has signal.
  * camera is roughly horizontal (zero pitch/roll). Tilt biases distance
    estimates by ~tan(tilt_deg) per metre — small for typical mounts.

Pipeline per annotated image:
  1. Undistort the fisheye to a PINHOLE crop (re-uses undistort.py's intrinsics).
  2. SegFormer-Cityscapes → road & terrain masks.
  3. For each image column with terrain pixels:
       - v_bot = lowest terrain pixel in the column (where grass meets road)
       - v_top = highest terrain pixel in the column
     Filter to: bottom adjacent to road, bottom below horizon, top below
     horizon, ground intersection within `max_dist_m`.
  4. Pinhole single-view metrology with vertical-object assumption:
       height = H * (v_bot - v_top) / (v_bot - cy)
     (derivation in the docstring of `column_height`)
  5. Per side: median height across kept columns.
  6. Aggregate per (height_class, confidence) bucket and report.

Run:
  python -m exp_height.src.single_view_metrology --per-class 100
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

EXP_ROOT = Path(__file__).resolve().parents[2]
EH_ROOT = EXP_ROOT / "exp_height"
sys.path.insert(0, str(EXP_ROOT))
from src import segmentation  # noqa: E402

ANNOT = Path("/home/henri/projects/MapillaryTest/MapillaryDatasetL+R-Annoted-v2.json")
IMAGES_DIR = Path("/home/henri/projects/MapillaryTest/images")
CROPS_DIR = Path("/home/henri/projects/MapillaryTest/crops")  # subdirs left/ right/
SRC_INTR = EXP_ROOT / "outputs" / "stage1_calib" / "intrinsics.json"


def load_undistort_maps(out_w=1280, out_h=960, hfov_deg=90.0):
    d = json.loads(SRC_INTR.read_text())["intrinsics"]
    K = np.array([[d["fx"], 0, d["cx"]], [0, d["fy"], d["cy"]], [0, 0, 1]], dtype=np.float64)
    D = np.array([[d["k1"]], [d["k2"]], [d["k3"]], [d["k4"]]], dtype=np.float64)
    src_w, src_h = int(d["width"]), int(d["height"])
    fx_new = out_w / (2.0 * np.tan(np.radians(hfov_deg) / 2.0))
    fy_new = fx_new
    K_new = np.array([[fx_new, 0, out_w / 2.0],
                      [0, fy_new, out_h / 2.0],
                      [0, 0, 1.0]], dtype=np.float64)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K_new, (out_w, out_h), cv2.CV_16SC2,
    )
    return map1, map2, (src_w, src_h), (out_w, out_h, fx_new, fy_new, out_w / 2.0, out_h / 2.0)


def build_roi_mask_in_undistorted(image_id: str, side: str,
                                  src_size: tuple, map1, map2,
                                  out_size: tuple) -> np.ndarray | None:
    """
    Crops are bottom-aligned half-images of the original fisheye:
        left:  original[(H_orig - h_crop):, 0:1352]
        right: original[(H_orig - h_crop):, 1352:2704]
    Non-black pixels in the crop are the labeled grass region. Build a full
    src-resolution mask from that, then warp through the same fisheye→PINHOLE
    remap so it aligns with the segmented image.
    """
    crop_path = CROPS_DIR / side / f"{image_id}.jpg"
    if not crop_path.exists():
        return None
    crop = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
    if crop is None:
        return None
    ch, cw = crop.shape[:2]
    src_w, src_h = src_size
    # build full-res mask in fisheye source coords
    mask_src = np.zeros((src_h, src_w), dtype=np.uint8)
    y0 = src_h - ch
    if side == "right":
        x0 = src_w - cw
    else:
        x0 = 0
    if y0 < 0 or x0 < 0:
        return None
    crop_mask = (crop.sum(axis=2) > 30).astype(np.uint8) * 255
    mask_src[y0:y0 + ch, x0:x0 + cw] = crop_mask
    # warp through fisheye undistort using same map (NEAREST so it stays binary)
    mask_und = cv2.remap(mask_src, map1, map2, interpolation=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return mask_und > 127


def load_annotations() -> list[dict]:
    """Returns list of {image_id, side, height, confidence}."""
    raw = json.loads(ANNOT.read_text())
    out = []
    for it in raw:
        if not it.get("annotations"):
            continue
        h, conf = None, None
        for r in it["annotations"][0].get("result", []):
            if r.get("from_name") == "height_class":
                h = r["value"]["choices"][0]
            if r.get("from_name") == "confidence":
                conf = r["value"]["choices"][0]
        if not h:
            continue
        out.append({
            "image_id": it["data"]["image_id"],
            "side": it["data"]["side"],
            "height": h,
            "confidence": conf,
        })
    return out


def column_heights(road_mask: np.ndarray, terrain_mask: np.ndarray,
                   intr: tuple, mount_h: float,
                   max_dist_m: float = 15.0,
                   min_v_below_cy_px: int = 60,
                   min_terrain_pixels_per_col: int = 4) -> list[dict]:
    """
    Apply pinhole single-view metrology per image column.

    Geometry: pinhole camera looks along +Z, image y axis points down. Camera
    is at height H above the road (Y=H in camera frame, Y axis down). For a
    pixel (u, v) below the principal point, the ray hits the ground plane at:
        Z_g = f * H / (v - cy)
        X_g = (u - cx) * Z_g / f  ;  Y_g = H
    For the top of a vertical object at the same (X_g, Z_g) projecting to
    pixel (u, v_top):
        Y_top = (v_top - cy) * Z_g / f
    Height of object above ground:
        h = H - Y_top = H * (v_bot - v_top) / (v_bot - cy)

    `min_v_below_cy_px` constrains the *bottom* of grass to be far enough below
    the principal point (i.e. close enough that the ground intersection is
    within `max_dist_m`).
    """
    H, W = road_mask.shape
    out_w, out_h, fx, fy, cx, cy = intr  # noqa: N806
    rows = np.arange(H)[:, None]  # (H, 1)
    # for each column find lowest (max v) and highest (min v) terrain pixel
    out = []
    for x in range(W):
        col_t = terrain_mask[:, x]
        if col_t.sum() < min_terrain_pixels_per_col:
            continue
        ys = np.where(col_t)[0]
        v_bot = int(ys.max())
        v_top = int(ys.min())
        # require bottom below horizon by a margin
        if (v_bot - cy) < min_v_below_cy_px:
            continue
        # require top below horizon too (else object pretends to extend above sky)
        if (v_top - cy) < 1.0:
            continue
        # require bottom adjacent to road or close to image bottom: terrain must
        # actually be standing on the ground we modeled, not floating.
        adj = False
        for dy in (1, 2, 3, 4, 5):
            yy = v_bot + dy
            if yy >= H:
                adj = True
                break
            if road_mask[yy, x]:
                adj = True
                break
        if not adj:
            continue
        # ground distance in metres
        Z_g = fy * mount_h / (v_bot - cy)
        if Z_g > max_dist_m:
            continue
        height_m = mount_h * (v_bot - v_top) / (v_bot - cy)
        # also a "thickness of mask" alternative (less sensitive to picking wrong v_bot)
        out.append({
            "u": x, "v_bot": v_bot, "v_top": v_top,
            "Z_g_m": float(Z_g), "height_m": float(height_m),
        })
    return out


def process_image(image_id: str, sides_needed: set[str],
                  map1, map2, src_size, intr, mount_h: float,
                  max_dist_m: float, save_overlay_to: Path | None = None) -> dict:
    """
    Compute per-side metrology for the requested sides, restricting the terrain
    mask to the labeler's painted ROI (the crop's non-black region).
    """
    im_path = IMAGES_DIR / f"{image_id}.jpg"
    if not im_path.exists():
        return {"image_id": image_id, "ok": False, "reason": "missing_image"}
    bgr = cv2.imread(str(im_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return {"image_id": image_id, "ok": False, "reason": "read_fail"}
    und_bgr = cv2.remap(bgr, map1, map2, interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT)
    und = cv2.cvtColor(und_bgr, cv2.COLOR_BGR2RGB)
    masks = segmentation.segment_classmap(und, ["road", "terrain", "vegetation"], device="cuda")

    out_size = und.shape[1::-1]  # (W, H)
    res = {"image_id": image_id, "ok": True}
    overlay_records: list[tuple[str, list[dict]]] = []
    for side in sides_needed:
        roi = build_roi_mask_in_undistorted(image_id, side, src_size, map1, map2, out_size)
        if roi is None:
            res[side] = {"n_cols": 0, "reason": "no_crop"}
            continue
        terrain_in_roi = masks["terrain"] & roi
        if terrain_in_roi.sum() < 50:
            res[side] = {"n_cols": 0, "reason": "empty_roi_terrain"}
            continue
        cols = column_heights(masks["road"], terrain_in_roi, intr, mount_h, max_dist_m=max_dist_m)
        if not cols:
            res[side] = {"n_cols": 0, "reason": "no_metrology_columns"}
            continue
        h = np.array([c["height_m"] for c in cols])
        z = np.array([c["Z_g_m"] for c in cols])
        res[side] = {
            "n_cols": int(len(cols)),
            "median_h_m": float(np.median(h)),
            "p25_h_m": float(np.percentile(h, 25)),
            "p75_h_m": float(np.percentile(h, 75)),
            "p90_h_m": float(np.percentile(h, 90)),
            "median_dist_m": float(np.median(z)),
        }
        overlay_records.append((side, cols))

    if save_overlay_to is not None and overlay_records:
        ov = und.copy()
        for side, cols in overlay_records:
            for c in cols:
                color = (255, 0, 0) if c["height_m"] > 0.6 else (
                        (255, 165, 0) if c["height_m"] > 0.25 else (0, 255, 0))
                cv2.line(ov, (c["u"], c["v_top"]), (c["u"], c["v_bot"]), color, 1)
        Image.fromarray(ov).save(save_overlay_to)
    return res


def summarize(records: list[dict]) -> dict:
    """records: list of {height (label), confidence, side, median_h_m, n_cols} flat rows."""
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in records:
        if r.get("median_h_m") is None:
            continue
        buckets[(r["height"], r["confidence"])].append(r["median_h_m"])
    out = {}
    for (h, c), vals in sorted(buckets.items()):
        a = np.array(vals)
        out[f"{h}__{c}"] = {
            "n": len(a),
            "mean_h_m": float(a.mean()),
            "median_h_m": float(np.median(a)),
            "p25_h_m": float(np.percentile(a, 25)),
            "p75_h_m": float(np.percentile(a, 75)),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mount-height-m", type=float, default=1.5)
    ap.add_argument("--max-dist-m", type=float, default=15.0)
    ap.add_argument("--per-class", type=int, default=100,
                    help="how many 'certo' annotations per class (B/M/A) to process")
    ap.add_argument("--save-overlays", type=int, default=10,
                    help="save overlays for the first N processed images per class")
    args = ap.parse_args()

    out_dir = EH_ROOT / "metrology"
    out_dir.mkdir(parents=True, exist_ok=True)
    ov_dir = out_dir / "overlays"
    ov_dir.mkdir(exist_ok=True)

    map1, map2, src_size, intr = load_undistort_maps()
    print(f"intrinsics (PINHOLE): out={intr[0]}x{intr[1]} fx={intr[2]:.1f} cy={intr[5]:.1f}")
    print(f"mount_h={args.mount_height_m} m, max_dist={args.max_dist_m} m")

    annots = load_annotations()
    # subset: certo only, balanced per class
    by_cls = defaultdict(list)
    for a in annots:
        if a["confidence"] == "certo":
            by_cls[a["height"]].append(a)
    sample = []
    for cls in ("Baixo", "Médio", "Alto"):
        sample.extend(by_cls[cls][: args.per_class])
    print(f"sampling {len(sample)} annotations: " +
          ", ".join(f"{c}={min(args.per_class, len(by_cls[c]))}" for c in ("Baixo", "Médio", "Alto")))

    # group annotations by image_id so we segment each image only once
    sides_per_image: dict[str, set[str]] = defaultdict(set)
    annots_per_image: dict[str, list[dict]] = defaultdict(list)
    for a in sample:
        sides_per_image[a["image_id"]].add(a["side"])
        annots_per_image[a["image_id"]].append(a)

    rows = []
    saved_overlays = defaultdict(int)
    image_ids = list(sides_per_image.keys())
    for i, iid in enumerate(image_ids):
        # decide overlay path based on the first label class for this image
        first = annots_per_image[iid][0]
        ov_path = None
        if saved_overlays[first["height"]] < args.save_overlays:
            ov_path = ov_dir / f"{first['height']}_{iid}.jpg"
            saved_overlays[first["height"]] += 1
        res = process_image(iid, sides_per_image[iid], map1, map2, src_size, intr,
                            args.mount_height_m, args.max_dist_m,
                            save_overlay_to=ov_path)
        if (i + 1) % 20 == 0:
            print(f"  processed {i+1}/{len(image_ids)} images")
        if not res.get("ok"):
            continue
        for a in annots_per_image[iid]:
            side_res = res.get(a["side"], {})
            rows.append({
                "image_id": iid,
                "side": a["side"],
                "height": a["height"],
                "confidence": a["confidence"],
                "median_h_m": side_res.get("median_h_m"),
                "p75_h_m": side_res.get("p75_h_m"),
                "p90_h_m": side_res.get("p90_h_m"),
                "n_cols": side_res.get("n_cols", 0),
                "median_dist_m": side_res.get("median_dist_m"),
            })

    (out_dir / "rows.json").write_text(json.dumps(rows, indent=2))
    summary = summarize(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== summary by (label, confidence) ===")
    for k, v in summary.items():
        print(f"  {k:20s} n={v['n']:4d}  mean={v['mean_h_m']:.3f}m  "
              f"p25/p50/p75 = {v['p25_h_m']:.3f} / {v['median_h_m']:.3f} / {v['p75_h_m']:.3f} m")

    # Crude rank correlation: Baixo < Médio < Alto?
    medians = {cls: summary.get(f"{cls}__certo", {}).get("median_h_m") for cls in ("Baixo", "Médio", "Alto")}
    print("\nclass medians (certo only):", medians)
    if all(v is not None for v in medians.values()):
        ord_ok = medians["Baixo"] < medians["Médio"] < medians["Alto"]
        print(f"ordering Baixo<Médio<Alto: {'YES' if ord_ok else 'NO'}")
    print(f"wrote {out_dir}/rows.json and summary.json; overlays under {ov_dir}")


if __name__ == "__main__":
    main()
