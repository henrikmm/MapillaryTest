"""
Stage 2 — per-frame height estimation (SfM-free).

This stage replaces the original stage 2 + 3 + 4. The minimum-viable design
(see ASSUMPTIONS.md A1, A8) needs only:

  * Camera intrinsics from stage 1 (one-shot OPENCV_FISHEYE calibration).
  * Per-pixel metric depth from Depth-Anything-V2 (zero-shot).
  * A SegFormer-Cityscapes mask for the `road` (ground) and `terrain` (grass)
    and `vegetation` (trees) classes.

For each frame:
  1. Load image, depth, segmentation.
  2. Lift every pixel to a 3D camera-frame point using
     P = depth * ray / ray_z (Z-along-axis depth convention).
  3. RANSAC-fit a plane to the `road` 3D points → ground plane (in cam frame).
  4. Camera height above plane = |offset|. Diagnostic; this is ALSO the
     SfM-free recovery of the mount height the user wanted to discover.
  5. For every pixel: signed distance to the ground plane along the plane
     normal = `height_m`. Heights are saved for terrain (grass) and vegetation
     (trees) separately so downstream analysis can pick the class it cares
     about — per the user's note, terrain is the actual grass target.

Outputs per frame:
    outputs/stage2_heights/<frame_id>/
        plane.json        # plane normal + offset, camera height, n_inliers
        heights.npz       # height_m (HxW float32), valid_mask, terrain_mask, vegetation_mask, road_mask
        overlay.png       # RGB + height heatmap (only if diagnostics.save_overlays)
        stats.json        # per-class height percentiles + counts

A run-level summary at outputs/stage2_heights/_summary.json aggregates camera
heights across frames so we can check cross-frame consistency without ever
having had ground truth.

Run:
    python -m src.stage2_per_frame_heights --config config.yaml --limit 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import depth as depth_mod
from . import segmentation
from .common import Config, load_config, setup_logging, write_json
from .geometry import (
    fit_plane_ransac,
    lift_to_3d,
    load_intrinsics,
    pixel_grid_to_rays,
)


def _orient_plane_against_camera(normal: np.ndarray, offset: float) -> tuple[np.ndarray, float]:
    """
    Flip the plane so the camera (at origin) sits on the positive-normal side.
    Then `offset` becomes negative (camera is at signed distance -offset above
    the plane); we report camera_height_m = -offset > 0.
    """
    if offset > 0:  # camera at origin: 0 - offset = -offset; want this > 0 → offset < 0
        return -normal, -offset
    return normal, offset


def _make_overlay(image_rgb: np.ndarray, heights: np.ndarray, mask: np.ndarray,
                  alpha: float = 0.55, vmin: float = 0.0, vmax: float = 2.0) -> np.ndarray:
    """Blend a height-coloured heatmap on top of the RGB image, only where mask is True."""
    try:
        import matplotlib.cm as cm
        h_clipped = np.clip(heights, vmin, vmax)
        norm = (h_clipped - vmin) / max(vmax - vmin, 1e-6)
        cmap = (cm.viridis(norm)[..., :3] * 255).astype(np.uint8)
    except ImportError:
        # Fallback: simple two-colour blend (red where high, green where low) without matplotlib.
        norm = np.clip((heights - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
        cmap = np.stack([(norm * 255).astype(np.uint8),
                         ((1 - norm) * 255).astype(np.uint8),
                         np.zeros_like(heights, dtype=np.uint8)], axis=-1)
    out = image_rgb.copy()
    m3 = mask[..., None]
    out = np.where(m3, (alpha * cmap + (1 - alpha) * image_rgb).astype(np.uint8), out)
    return out


def _per_column_thickness(height_m: np.ndarray, mask: np.ndarray,
                          depth: np.ndarray | None = None,
                          min_pixels_per_col: int = 8,
                          depth_band_m: float = 2.0,
                          max_depth_m: float = 15.0,
                          lo_pct: float = 10.0,
                          hi_pct: float = 90.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-column canopy thickness along the ground-plane normal.

    Naive max(h)-min(h) over a column inflates badly because a single image
    column samples many camera distances on a fisheye (foreground grass strip
    + distant hillside) and one outlier dominates. We restrict per column to:
      * pixels closer than `max_depth_m` (drop the far hill that breaks the
        "one column = one ground patch" assumption),
      * pixels within a `depth_band_m` window around the column's median depth
        (keep only the locally-coherent grass patch),
    then report `pXX(height) - pYY(height)` (robust to single-pixel outliers).

    Scalar depth-model bias inflates both endpoints by ~the same factor inside
    the band, so the *difference* survives much better than either absolute
    height alone.

    Returns (thickness_per_col, valid_cols) — both length-W arrays.
    """
    H, W = height_m.shape
    thick = np.full(W, np.nan, dtype=np.float32)
    valid = np.zeros(W, dtype=bool)
    for x in range(W):
        col_mask = mask[:, x]
        if col_mask.sum() < min_pixels_per_col:
            continue
        h_col = height_m[col_mask, x]
        if depth is not None:
            d_col = depth[col_mask, x]
            keep = np.isfinite(d_col) & (d_col > 0) & (d_col < max_depth_m)
            if keep.sum() < min_pixels_per_col:
                continue
            d_kept = d_col[keep]
            h_kept = h_col[keep]
            d_med = float(np.median(d_kept))
            band = np.abs(d_kept - d_med) <= depth_band_m
            if band.sum() < min_pixels_per_col:
                continue
            h_band = h_kept[band]
        else:
            h_band = h_col
        if not np.isfinite(h_band).any():
            continue
        lo, hi = np.percentile(h_band, [lo_pct, hi_pct])
        thick[x] = hi - lo
        valid[x] = True
    return thick, valid


def _summarise_thickness(name: str, height_m: np.ndarray, mask: np.ndarray,
                         depth: np.ndarray | None = None) -> dict:
    thick, valid = _per_column_thickness(height_m, mask, depth=depth)
    if valid.sum() < 5:
        return {"class": name, "n_columns": int(valid.sum()), "ok": False}
    vals = thick[valid]
    pct = np.percentile(vals, [5, 25, 50, 75, 95]).tolist()
    return {
        "class": name,
        "n_columns": int(valid.sum()),
        "ok": True,
        "min_m": float(vals.min()),
        "max_m": float(vals.max()),
        "mean_m": float(vals.mean()),
        "p05_m": pct[0], "p25_m": pct[1], "p50_m": pct[2], "p75_m": pct[3], "p95_m": pct[4],
    }


def _summarise_heights(name: str, h: np.ndarray, mask: np.ndarray) -> dict:
    if mask.sum() < 10:
        return {"class": name, "n_pixels": int(mask.sum()), "ok": False}
    vals = h[mask]
    pct = np.percentile(vals, [5, 25, 50, 75, 95]).tolist()
    return {
        "class": name,
        "n_pixels": int(mask.sum()),
        "ok": True,
        "min_m": float(vals.min()),
        "max_m": float(vals.max()),
        "mean_m": float(vals.mean()),
        "p05_m": pct[0], "p25_m": pct[1], "p50_m": pct[2], "p75_m": pct[3], "p95_m": pct[4],
    }


def process_frame(image_path: Path, intr: dict, cfg: Config,
                  out_dir: Path, log) -> dict:
    """Run the full per-frame pipeline; return a small dict for the run summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fid = image_path.stem
    image = np.asarray(Image.open(image_path).convert("RGB"))
    H, W = image.shape[:2]

    # 1. depth + segmentation
    log.info("[%s] depth + segmentation (HxW=%dx%d)", fid, H, W)
    depth = depth_mod.predict_depth(image, device=cfg["depth"]["device"])
    masks = segmentation.segment_classmap(
        image, ["road", "terrain", "vegetation", "sky"],
        device=cfg["depth"]["device"],
    )

    # 2. lift to 3D
    rays = pixel_grid_to_rays(intr, H, W)
    # depth_convention="ray" works much better than "z" on this fisheye:
    # Depth Anything V2 was trained on perspective images, so its output for an
    # off-axis fisheye pixel is closer to the radial distance than the optical-Z
    # distance. Empirically (see commit notes) "ray" produces 4× more RANSAC
    # inliers and a nearly-vertical ground-plane normal vs. "z".
    depth_convention = cfg.get("depth", {}).get("convention", "ray")
    pts = lift_to_3d(rays, depth, depth_convention=depth_convention)  # (H, W, 3) cam-frame

    # 3. ground plane from road pixels
    gp_cfg = cfg["ground_plane"]
    road = masks["road"]
    valid_road = road & np.isfinite(depth) & (depth > 0)
    if valid_road.sum() < gp_cfg["min_inlier_points"]:
        log.warning("[%s] only %d road pixels — skipping", fid, int(valid_road.sum()))
        return {"frame": fid, "ok": False, "reason": f"insufficient_road_pixels:{int(valid_road.sum())}"}
    road_pts = pts[valid_road]
    # Diagnostic: what depth/3D values are we actually working with?
    rd = depth[valid_road]
    log.info("[%s] road depth (m) p10/p50/p90=%.2f/%.2f/%.2f  | "
             "road P_y (cam-down) p10/p50/p90=%.2f/%.2f/%.2f  | "
             "P_z p10/p50/p90=%.2f/%.2f/%.2f",
             fid,
             float(np.percentile(rd, 10)), float(np.percentile(rd, 50)), float(np.percentile(rd, 90)),
             float(np.percentile(road_pts[:, 1], 10)), float(np.percentile(road_pts[:, 1], 50)), float(np.percentile(road_pts[:, 1], 90)),
             float(np.percentile(road_pts[:, 2], 10)), float(np.percentile(road_pts[:, 2], 50)), float(np.percentile(road_pts[:, 2], 90)))
    # Cap to keep RANSAC fast.
    cap = 50_000
    if len(road_pts) > cap:
        idx = np.random.default_rng(0).choice(len(road_pts), cap, replace=False)
        road_pts = road_pts[idx]
    normal, offset, inliers = fit_plane_ransac(
        road_pts,
        threshold=gp_cfg["ransac_threshold_m"],
        iterations=gp_cfg["ransac_iterations"],
    )
    normal, offset = _orient_plane_against_camera(normal, offset)
    cam_height = -offset  # signed distance from origin to plane along +normal
    log.info("[%s] ground plane: cam_height=%.2f m, inliers=%d/%d, normal=[%.2f %.2f %.2f]",
             fid, cam_height, int(inliers.sum()), len(road_pts), *normal)

    # 4. per-pixel signed height above plane
    valid = np.isfinite(depth) & (depth > 0) & ~masks["sky"]
    height_m = (pts @ normal) - offset                # (H, W) signed
    height_m[~valid] = np.nan

    # 5. per-column canopy thickness (depth-bias-robust complement to absolute height)
    terrain_valid = masks["terrain"] & valid
    veget_valid = masks["vegetation"] & valid
    thick_terrain, _ = _per_column_thickness(height_m, terrain_valid, depth=depth)
    thick_veget, _   = _per_column_thickness(height_m, veget_valid, depth=depth)
    # Broadcast per-column thickness back to a (H, W) map for overlay rendering.
    thick_terrain_map = np.broadcast_to(thick_terrain, height_m.shape).copy()
    thick_terrain_map[~terrain_valid] = np.nan

    # 6. save
    np.savez_compressed(
        out_dir / "heights.npz",
        height_m=height_m.astype(np.float32),
        thickness_terrain_per_col=thick_terrain.astype(np.float32),
        thickness_vegetation_per_col=thick_veget.astype(np.float32),
        valid=valid,
        road=masks["road"],
        terrain=masks["terrain"],
        vegetation=masks["vegetation"],
        sky=masks["sky"],
    )
    write_json(out_dir / "plane.json", {
        "frame": fid,
        "normal_camframe": normal.tolist(),
        "offset_m": float(offset),
        "camera_height_m": float(cam_height),
        "n_road_inliers": int(inliers.sum()),
        "n_road_candidates": int(len(road_pts)),
    })

    stats = {
        "frame": fid,
        "camera_height_m": float(cam_height),
        "classes": {
            "terrain":    _summarise_heights("terrain",    height_m, terrain_valid),
            "vegetation": _summarise_heights("vegetation", height_m, veget_valid),
            "road":       _summarise_heights("road",       height_m, masks["road"] & valid),
        },
        "thickness_per_column": {
            "terrain":    _summarise_thickness("terrain",    height_m, terrain_valid, depth=depth),
            "vegetation": _summarise_thickness("vegetation", height_m, veget_valid,   depth=depth),
        },
    }
    write_json(out_dir / "stats.json", stats)

    # 7. overlays — absolute height AND per-column thickness, side-by-side artefacts.
    if cfg.get("diagnostics", {}).get("save_overlays", True):
        alpha = cfg["diagnostics"].get("overlay_alpha", 0.55)
        if terrain_valid.any():
            ov_h = _make_overlay(image, height_m, terrain_valid,
                                 alpha=alpha,
                                 vmin=cfg["vegetation_height"]["min_height_m"],
                                 vmax=cfg["vegetation_height"]["max_height_m"])
            Image.fromarray(ov_h).save(out_dir / "overlay_terrain_height.png")
            # Thickness range tends to be smaller than absolute height; give it a tighter scale.
            ov_t = _make_overlay(image, thick_terrain_map, terrain_valid,
                                 alpha=alpha, vmin=0.0, vmax=1.5)
            Image.fromarray(ov_t).save(out_dir / "overlay_terrain_thickness.png")

    return {"frame": fid, "ok": True, "camera_height_m": float(cam_height),
            "terrain_height_p50_m":     stats["classes"]["terrain"].get("p50_m"),
            "vegetation_height_p50_m":  stats["classes"]["vegetation"].get("p50_m"),
            "terrain_thickness_p50_m":    stats["thickness_per_column"]["terrain"].get("p50_m"),
            "vegetation_thickness_p50_m": stats["thickness_per_column"]["vegetation"].get("p50_m"),
            }


def run(cfg: Config, limit: int | None = None, frames: list[str] | None = None) -> None:
    log = setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    intr_path = cfg.outputs_dir / "stage1_calib" / "intrinsics.json"
    if not intr_path.exists():
        raise SystemExit(f"missing {intr_path} — run stage 1 (calibrate) first.")
    intr = load_intrinsics(intr_path)
    log.info("intrinsics: model=%s fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
             intr["model"], intr["fx"], intr["fy"], intr["cx"], intr["cy"])

    images_dir = cfg.path("images_dir")
    out_root = cfg.outputs_dir / "stage2_heights"
    out_root.mkdir(parents=True, exist_ok=True)

    # Pick frames: explicit list, or take the pilot[0] sub-sequence.
    if frames:
        frame_ids = frames
    else:
        manifest = cfg.outputs_dir / "stage0_manifest" / "pilot_sequences.json"
        with manifest.open() as f:
            seq = json.load(f)["pilot"][0]
        frame_ids = [f["id"] for f in seq["frames"]]
    if limit is not None:
        frame_ids = frame_ids[:limit]

    log.info("processing %d frames -> %s", len(frame_ids), out_root)
    summary = []
    for fid in frame_ids:
        image_path = images_dir / f"{fid}.jpg"
        if not image_path.exists():
            log.warning("missing image %s", image_path)
            continue
        try:
            summary.append(process_frame(image_path, intr, cfg, out_root / fid, log))
        except Exception as e:  # noqa: BLE001
            log.exception("[%s] failed", fid)
            summary.append({"frame": fid, "ok": False, "error": str(e)})

    # Cross-frame camera-height consistency = the key sanity check for the
    # SfM-free mount-height recovery (ASSUMPTIONS.md A1).
    heights = [r["camera_height_m"] for r in summary if r.get("ok") and "camera_height_m" in r]
    cross = {}
    if heights:
        h = np.array(heights)
        cross = {
            "n": len(heights),
            "median_m": float(np.median(h)),
            "mean_m":   float(h.mean()),
            "std_m":    float(h.std()),
            "min_m":    float(h.min()),
            "max_m":    float(h.max()),
            "iqr_m":    float(np.percentile(h, 75) - np.percentile(h, 25)),
        }
        log.info("cross-frame camera height: median=%.2fm  std=%.2fm  IQR=%.2fm  (n=%d)",
                 cross["median_m"], cross["std_m"], cross["iqr_m"], cross["n"])
    write_json(out_root / "_summary.json", {
        "frames": summary,
        "cross_frame_camera_height_m": cross,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--frame", action="append", default=None,
                    help="explicit frame id (repeatable); overrides pilot selection")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run(cfg, limit=args.limit, frames=args.frame)


if __name__ == "__main__":
    main()
