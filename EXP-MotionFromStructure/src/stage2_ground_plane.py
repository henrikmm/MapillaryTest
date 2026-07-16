"""
Stage 2 — ground plane estimation.

Inputs:
    outputs/stage1_sfm/<seq>/reconstruction.json   (SfM points + poses, metric)
    images/                                         (for segmentation)

Output:
    outputs/stage2_ground/<seq>/ground_plane.json
        {
            "plane": {"normal": [nx, ny, nz], "offset": d},   # n.x = d
            "n_inliers": int,
            "camera_height_m": float,                # mean cam height above plane
            "sanity": {
                "expected_m": 1.5, "tolerance_frac": 0.30,
                "passes": bool, "deviation_frac": float
            }
        }

Run:
    python -m src.stage2_ground_plane --config config.yaml [--seq <id>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common import Config, load_config, setup_logging, write_json
from . import segmentation


def _ransac_plane(points: np.ndarray, threshold: float, iters: int, rng: np.random.Generator):
    """
    Plain plane RANSAC. `points` is (N, 3). Returns (normal[3], offset, inlier_mask).
    Plane equation: normal · x = offset.
    """
    best_inliers = np.zeros(len(points), dtype=bool)
    best_plane = (np.array([0.0, 0.0, 1.0]), 0.0)
    n = len(points)
    if n < 3:
        return best_plane[0], best_plane[1], best_inliers
    for _ in range(iters):
        idx = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = points[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        nrm = np.linalg.norm(normal)
        if nrm < 1e-9:
            continue
        normal = normal / nrm
        offset = float(normal @ p0)
        dist = np.abs(points @ normal - offset)
        inliers = dist < threshold
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_plane = (normal, offset)
    # Refit on inliers via least squares (centroid + SVD).
    if best_inliers.sum() >= 3:
        pts = points[best_inliers]
        c = pts.mean(axis=0)
        _, _, vh = np.linalg.svd(pts - c, full_matrices=False)
        normal = vh[-1]
        offset = float(normal @ c)
        best_plane = (normal, offset)
    return best_plane[0], best_plane[1], best_inliers


def _orient_plane_up(normal: np.ndarray, offset: float, cam_centers: np.ndarray) -> tuple[np.ndarray, float]:
    """Flip the plane so cameras are on the positive-normal side (i.e. normal points "up")."""
    side = np.sign(np.median(cam_centers @ normal - offset))
    if side < 0:
        return -normal, -offset
    return normal, offset


def _camera_centers_from_recon(recon: list) -> np.ndarray:
    out = []
    for r in recon:
        for shot in r.get("shots", {}).values():
            # OpenSfM shot stores rotation (axis-angle) and translation in world frame.
            R = _axis_angle_to_R(np.array(shot["rotation"], dtype=float))
            t = np.array(shot["translation"], dtype=float)
            # camera center C = -R^T t
            out.append(-R.T @ t)
    return np.asarray(out) if out else np.zeros((0, 3))


def _axis_angle_to_R(rvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-12:
        return np.eye(3)
    k = rvec / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def _points_from_recon(recon: list) -> np.ndarray:
    out = []
    for r in recon:
        for p in r.get("points", {}).values():
            out.append(p["coordinates"])
    return np.asarray(out, dtype=float) if out else np.zeros((0, 3))


def estimate_for_sequence(cfg: Config, seq_id: str) -> dict:
    log = setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    gp = cfg["ground_plane"]
    recon_path = cfg.outputs_dir / "stage1_sfm" / seq_id / "reconstruction.json"
    if not recon_path.exists():
        raise FileNotFoundError(f"missing reconstruction for {seq_id}: {recon_path}")
    with recon_path.open() as f:
        recon = json.load(f)

    all_points = _points_from_recon(recon)
    cam_centers = _camera_centers_from_recon(recon)
    log.info("seq %s: %d points, %d cameras", seq_id, len(all_points), len(cam_centers))

    # NOTE: a fuller implementation projects each point into the source images,
    # filters by the road/terrain segmentation mask, and runs RANSAC only on those.
    # For now we run RANSAC on the bottom 30% of points by altitude as a coarse
    # ground-biased heuristic — a usable baseline until segmentation is wired up.
    if len(all_points) < gp["min_inlier_points"]:
        return {"sequence": seq_id, "ok": False, "error": "too few points"}

    z_thresh = np.percentile(all_points[:, 2], 30)
    candidates = all_points[all_points[:, 2] <= z_thresh]
    rng = np.random.default_rng(0)
    normal, offset, inliers = _ransac_plane(
        candidates, gp["ransac_threshold_m"], gp["ransac_iterations"], rng
    )
    normal, offset = _orient_plane_up(normal, offset, cam_centers)
    cam_heights = cam_centers @ normal - offset
    cam_height = float(np.median(cam_heights))

    # Camera height is a discovery target — see ASSUMPTIONS.md A1. Only run the
    # absolute-height sanity check if the user opted in by setting an expected value.
    expected = gp.get("expected_camera_height_m")
    if expected is None:
        sanity = {"expected_m": None, "passes": None,
                  "note": "absolute height check disabled; cross-sequence consistency check happens in stage 5"}
    else:
        tol = gp["tolerance_frac"]
        deviation = abs(cam_height - expected) / max(expected, 1e-6)
        sanity = {
            "expected_m": expected, "tolerance_frac": tol,
            "deviation_frac": deviation, "passes": bool(deviation <= tol),
        }

    result = {
        "sequence": seq_id,
        "ok": bool(inliers.sum() >= gp["min_inlier_points"]),
        "plane": {"normal": normal.tolist(), "offset": offset},
        "n_inliers": int(inliers.sum()),
        "n_candidates": int(len(candidates)),
        "camera_height_m": cam_height,
        "sanity": sanity,
        "note": (
            "ground-bias by altitude percentile only; once segmentation is wired "
            "(see ASSUMPTIONS.md A6) restrict RANSAC to road/terrain pixels."
        ),
    }
    out_dir = cfg.outputs_dir / "stage2_ground" / seq_id
    write_json(out_dir / "ground_plane.json", result)
    log.info("seq %s: cam_height=%.2fm  sanity_passes=%s", seq_id, cam_height,
             result["sanity"].get("passes"))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--seq", action="append", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    manifest = cfg.outputs_dir / "stage0_manifest" / "pilot_sequences.json"
    with manifest.open() as f:
        seqs = [s["sequence"] for s in json.load(f)["pilot"]]
    if args.seq:
        seqs = [s for s in seqs if s in set(args.seq)]
    for sid in seqs:
        estimate_for_sequence(cfg, sid)


if __name__ == "__main__":
    main()
