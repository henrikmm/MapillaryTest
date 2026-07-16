"""
Grass-height estimation from the metric SfM reconstruction.

Pipeline:
  1. Load reconstruction.json (cameras + 3D points, in metres).
  2. Run SegFormer-Cityscapes once per registered image (undistorted PINHOLE).
  3. For every 3D point, project into every image that saw it, look up the
     class label at the 2D pixel. Majority vote across views → point class.
  4. RANSAC-fit a plane to all road-labelled 3D points → metric ground plane.
  5. Height of each terrain (grass) point = signed distance to plane along
     the plane normal.
  6. Report camera-mount-height (median of per-camera height-above-plane),
     terrain-height percentiles, and a per-frame overlay (terrain points
     coloured by height) for QA.

Outputs (under exp_height/heights/):
  ground_plane.json
  point_classes.json    — id -> {class, height_m, n_views, ...}
  per_frame.json        — per-frame stats
  overlay/<fid>.jpg     — RGB + projected terrain points coloured by height

Run:
  python -m exp_height.src.heights
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

EXP_ROOT = Path(__file__).resolve().parents[2]
EH_ROOT = EXP_ROOT / "exp_height"
sys.path.insert(0, str(EXP_ROOT))
from src import segmentation  # noqa: E402

CITY_ROAD = 0
CITY_VEG = 8
CITY_TERRAIN = 9
CITY_SKY = 10


def fit_plane_ransac(pts: np.ndarray, threshold: float, iterations: int,
                     min_inliers: int):
    """Plane n·x + d = 0 (||n||=1). Returns (n, d, inliers_mask)."""
    rng = np.random.default_rng(0)
    n_pts = len(pts)
    if n_pts < 3:
        raise ValueError("need >= 3 points")
    best_inliers = np.zeros(n_pts, dtype=bool)
    best_n = None
    best_d = 0.0
    for _ in range(iterations):
        idx = rng.choice(n_pts, 3, replace=False)
        p0, p1, p2 = pts[idx]
        v1, v2 = p1 - p0, p2 - p0
        n = np.cross(v1, v2)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        d = -float(n @ p0)
        dist = np.abs(pts @ n + d)
        inl = dist < threshold
        if inl.sum() > best_inliers.sum():
            best_inliers = inl
            best_n = n
            best_d = d
    if best_inliers.sum() < min_inliers:
        return best_n, best_d, best_inliers
    # SVD refit on the inlier set
    P = pts[best_inliers]
    centroid = P.mean(0)
    _, _, Vt = np.linalg.svd(P - centroid)
    n = Vt[-1]
    n /= np.linalg.norm(n)
    d = -float(n @ centroid)
    dist = np.abs(pts @ n + d)
    inl = dist < threshold
    return n, d, inl


def orient_plane_against_cameras(n, d, cam_centers: np.ndarray):
    """Flip plane so cameras are on the +n side; report camera height as +d_signed."""
    signed = cam_centers @ n + d  # positive if on +n side
    if np.median(signed) < 0:
        return -n, -d
    return n, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ransac-threshold-m", type=float, default=0.30)
    ap.add_argument("--ransac-iterations", type=int, default=2000)
    ap.add_argument("--min-road-inliers", type=int, default=30)
    ap.add_argument("--min-views-per-point", type=int, default=2)
    ap.add_argument("--save-overlays", action="store_true", default=True)
    ap.add_argument("--max-height-m", type=float, default=3.0)
    args = ap.parse_args()

    out_dir = EH_ROOT / "heights"
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = out_dir / "overlay"
    overlay_dir.mkdir(exist_ok=True)

    rec = json.loads((EH_ROOT / "sfm" / "reconstruction.json").read_text())
    intr = rec["intrinsics_used"]
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    W, H = intr["width"], intr["height"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    # ---- index cameras and points -----------------------------------------
    images = rec["images"]                # list[dict]
    # Reconstruct image-id mapping: COLMAP track elements use image_id, not name.
    # The reconstruction.json was exported in image-id order from rec.images.values(),
    # so position-in-list is NOT image_id. We need to reload via colmap to map ids,
    # but a simpler path: use the track elements that index INTO our images list
    # would need the original mapping. Instead: re-derive from colmap workspace.
    import pycolmap
    rec_obj = pycolmap.Reconstruction(EH_ROOT / "sfm" / "colmap_project" / "sparse" / "0")
    # apply the same Sim3d alignment (saved into reconstruction.json via rec.transform earlier).
    # Easier: reconstruct from exported file. Build per-image data:
    image_by_name: dict[str, dict] = {im["name"]: im for im in images}

    print(f"images: {len(images)}, points: {len(rec['points'])}")

    # ---- segment every registered image once ------------------------------
    seg_by_name: dict[str, np.ndarray] = {}     # name -> (H, W) class id array
    rgb_by_name: dict[str, np.ndarray] = {}
    img_dir = EH_ROOT / "images"
    for im in images:
        name = Path(im["name"]).name  # strip any path prefix colmap baked in
        im["_basename"] = name
        im_path = img_dir / name
        if not im_path.exists():
            continue
        rgb = np.asarray(Image.open(im_path).convert("RGB"))
        rgb_by_name[name] = rgb
        # ask segformer for the raw label map by requesting all four classes individually
        masks = segmentation.segment_classmap(rgb, ["road", "terrain", "vegetation", "sky"], device="cuda")
        # Build single class map from masks (priority: terrain > vegetation > road > sky > other)
        labels = np.full(rgb.shape[:2], -1, dtype=np.int8)
        labels[masks["sky"]]        = CITY_SKY
        labels[masks["road"]]       = CITY_ROAD
        labels[masks["vegetation"]] = CITY_VEG
        labels[masks["terrain"]]    = CITY_TERRAIN
        seg_by_name[name] = labels
        print(f"  segmented {name}")

    # ---- classify each 3D point by majority vote across its track ---------
    pt_records = {p["id"]: p for p in rec["points"]}
    pt_class: dict[int, str] = {}
    pt_class_counts: dict[int, dict[str, int]] = {}
    LABEL_NAME = {CITY_ROAD: "road", CITY_TERRAIN: "terrain",
                  CITY_VEG: "vegetation", CITY_SKY: "sky"}

    # We need per-image image_id -> name for the track. Use colmap reconstruction.
    img_id_to_name = {iid: Path(img.name).name for iid, img in rec_obj.images.items()}

    # Project points using post-alignment poses (from our exported R_wc/t_wc).
    pose_by_name = {}
    for im in images:
        Rwc = np.array(im["R_wc"])
        twc = np.array(im["t_wc"])
        Rcw = Rwc.T
        tcw = -Rcw @ twc
        pose_by_name[im["_basename"]] = (Rcw, tcw)

    for pt_id, pt in pt_records.items():
        Xw = np.array(pt["xyz"], dtype=np.float64)
        votes = []
        for image_id, _p2_idx in pt["track"]:
            name = img_id_to_name.get(image_id)
            if name is None or name not in pose_by_name or name not in seg_by_name:
                continue
            Rcw, tcw = pose_by_name[name]
            Xc = Rcw @ Xw + tcw
            if Xc[2] <= 0.05:
                continue
            u = fx * Xc[0] / Xc[2] + cx
            v = fy * Xc[1] / Xc[2] + cy
            ui, vi = int(round(u)), int(round(v))
            if 0 <= ui < W and 0 <= vi < H:
                lab = int(seg_by_name[name][vi, ui])
                if lab in LABEL_NAME:
                    votes.append(LABEL_NAME[lab])
        if not votes or len(votes) < args.min_views_per_point:
            continue
        c = Counter(votes)
        cls, _ = c.most_common(1)[0]
        pt_class[pt_id] = cls
        pt_class_counts[pt_id] = dict(c)

    by_class = Counter(pt_class.values())
    print(f"point classification: {dict(by_class)} (of {len(pt_records)} total)")

    # ---- fit ground plane to road points ----------------------------------
    road_pts = np.array([pt_records[i]["xyz"] for i, c in pt_class.items() if c == "road"], dtype=np.float64)
    if len(road_pts) < args.min_road_inliers:
        raise SystemExit(f"only {len(road_pts)} road points — need ≥ {args.min_road_inliers}")
    n, d, inl = fit_plane_ransac(road_pts, args.ransac_threshold_m, args.ransac_iterations,
                                 args.min_road_inliers)
    cam_centers = np.array([np.array(im["t_wc"]) for im in images])
    n, d = orient_plane_against_cameras(n, d, cam_centers)
    cam_heights = cam_centers @ n + d
    print(f"ground plane: n={n.tolist()} d={d:.3f}  road_inliers={int(inl.sum())}/{len(road_pts)}")
    print(f"camera heights above plane (m): median={np.median(cam_heights):.2f} "
          f"mean={cam_heights.mean():.2f} std={cam_heights.std():.2f} "
          f"min={cam_heights.min():.2f} max={cam_heights.max():.2f}")

    # ---- compute per-point height above plane -----------------------------
    pt_heights = {}
    for pid, cls in pt_class.items():
        Xw = np.array(pt_records[pid]["xyz"])
        h = float(Xw @ n + d)
        pt_heights[pid] = h

    # Terrain heights (the grass)
    t_h = np.array([pt_heights[i] for i, c in pt_class.items() if c == "terrain"])
    v_h = np.array([pt_heights[i] for i, c in pt_class.items() if c == "vegetation"])

    def summary(arr: np.ndarray):
        if len(arr) == 0:
            return None
        pct = np.percentile(arr, [5, 25, 50, 75, 95]).tolist()
        return {
            "n": int(len(arr)),
            "min_m": float(arr.min()), "max_m": float(arr.max()),
            "mean_m": float(arr.mean()),
            "p05_m": pct[0], "p25_m": pct[1], "p50_m": pct[2], "p75_m": pct[3], "p95_m": pct[4],
        }

    terrain_summary = summary(t_h)
    veget_summary = summary(v_h)
    print(f"terrain (grass) heights (m): {terrain_summary}")
    print(f"vegetation (trees) heights (m): {veget_summary}")

    # ---- save artefacts ---------------------------------------------------
    (out_dir / "ground_plane.json").write_text(json.dumps({
        "normal": n.tolist(),
        "offset_m": d,
        "ransac_threshold_m": args.ransac_threshold_m,
        "n_road_total": int(len(road_pts)),
        "n_road_inliers": int(inl.sum()),
        "camera_height_m": {
            "n": len(cam_heights),
            "median": float(np.median(cam_heights)),
            "mean": float(cam_heights.mean()),
            "std": float(cam_heights.std()),
            "min": float(cam_heights.min()),
            "max": float(cam_heights.max()),
            "per_frame": [{"name": im["name"], "height_m": float(h)}
                          for im, h in zip(images, cam_heights)],
        },
        "terrain_height_m": terrain_summary,
        "vegetation_height_m": veget_summary,
        "by_class_n": dict(by_class),
    }, indent=2))

    (out_dir / "point_classes.json").write_text(json.dumps([
        {"id": pid, "class": cls, "height_m": pt_heights[pid],
         "votes": pt_class_counts[pid],
         "xyz": pt_records[pid]["xyz"]}
        for pid, cls in pt_class.items()
    ], indent=2))

    # ---- per-frame overlays: project terrain points coloured by height ----
    if args.save_overlays:
        try:
            import matplotlib.cm as cm
            cmap_fn = lambda v: (cm.viridis(np.clip(v, 0, args.max_height_m) / args.max_height_m)[:3] * 255).astype(np.uint8)
        except ImportError:
            cmap_fn = lambda v: np.array([int(255 * np.clip(v / args.max_height_m, 0, 1)),
                                          int(255 * (1 - np.clip(v / args.max_height_m, 0, 1))), 0], dtype=np.uint8)

        for im in images:
            name = im["_basename"]
            if name not in rgb_by_name:
                continue
            rgb = rgb_by_name[name].copy()
            Rcw, tcw = pose_by_name[name]
            terrain_pts_drawn = 0
            for pid, cls in pt_class.items():
                if cls != "terrain":
                    continue
                Xw = np.array(pt_records[pid]["xyz"])
                Xc = Rcw @ Xw + tcw
                if Xc[2] <= 0.05:
                    continue
                u = fx * Xc[0] / Xc[2] + cx
                v = fy * Xc[1] / Xc[2] + cy
                ui, vi = int(round(u)), int(round(v))
                if not (0 <= ui < W and 0 <= vi < H):
                    continue
                col = cmap_fn(pt_heights[pid])
                # 5x5 blob
                y0, y1 = max(0, vi - 2), min(H, vi + 3)
                x0, x1 = max(0, ui - 2), min(W, ui + 3)
                rgb[y0:y1, x0:x1] = col
                terrain_pts_drawn += 1
            Image.fromarray(rgb).save(overlay_dir / name)

    # ---- per-frame summary -----------------------------------------------
    per_frame = []
    for im in images:
        name = im["_basename"]
        Rcw, tcw = pose_by_name[name]
        # collect terrain heights for points visible in this frame
        h_here = []
        for pid, cls in pt_class.items():
            if cls != "terrain":
                continue
            Xw = np.array(pt_records[pid]["xyz"])
            Xc = Rcw @ Xw + tcw
            if Xc[2] <= 0.05:
                continue
            u = fx * Xc[0] / Xc[2] + cx
            v = fy * Xc[1] / Xc[2] + cy
            if 0 <= int(round(u)) < W and 0 <= int(round(v)) < H:
                h_here.append(pt_heights[pid])
        h_here = np.array(h_here) if h_here else np.array([])
        per_frame.append({
            "name": name,
            "frame_id": im["frame_id"],
            "camera_height_m": float(np.array(im["t_wc"]) @ n + d),
            "n_terrain_pts_visible": int(len(h_here)),
            "terrain_p50_m": float(np.median(h_here)) if len(h_here) else None,
            "terrain_p90_m": float(np.percentile(h_here, 90)) if len(h_here) else None,
        })
    (out_dir / "per_frame.json").write_text(json.dumps(per_frame, indent=2))
    print(f"wrote {out_dir}/ground_plane.json, point_classes.json, per_frame.json")
    print(f"wrote {len(images)} overlays to {overlay_dir}")


if __name__ == "__main__":
    main()
