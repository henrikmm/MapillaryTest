"""
Camera-frame geometry helpers for OPENCV_FISHEYE intrinsics.

OPENCV_FISHEYE projection (matches COLMAP and OpenCV cv2.fisheye):
    Given a 3D point P = (X, Y, Z) in camera frame (Z forward),
        a = X / Z, b = Y / Z
        r = sqrt(a^2 + b^2)
        theta = atan(r)
        theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
        x' = (theta_d / r) * a   (or x' = a if r == 0)
        y' = (theta_d / r) * b
        u = fx * x' + cx
        v = fy * y' + cy

The inverse (pixel → unit ray) requires solving for theta given theta_d, which
has no closed form — we use Newton iteration.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_intrinsics(path: Path) -> dict:
    with Path(path).open() as f:
        d = json.load(f)
    if d.get("ok") and "intrinsics" in d:
        return d["intrinsics"]
    return d  # already an intrinsics dict


def _solve_theta(theta_d: np.ndarray, k: tuple[float, float, float, float],
                 iters: int = 10) -> np.ndarray:
    """Newton-iterate theta from theta_d using the OPENCV_FISHEYE distortion polynomial."""
    k1, k2, k3, k4 = k
    theta = theta_d.copy()
    for _ in range(iters):
        t2 = theta * theta
        t4 = t2 * t2
        t6 = t4 * t2
        t8 = t4 * t4
        f  = theta * (1 + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8) - theta_d
        df = 1 + 3 * k1 * t2 + 5 * k2 * t4 + 7 * k3 * t6 + 9 * k4 * t8
        theta = theta - f / np.maximum(df, 1e-12)
    return theta


def pixel_grid_to_rays(intr: dict, height: int, width: int) -> np.ndarray:
    """
    Build an (H, W, 3) array of unit camera-frame ray directions for every pixel.

    Convention: camera frame has +X right, +Y down, +Z forward (OpenCV / COLMAP).
    Returned vectors are unit length.
    """
    fx, fy = float(intr["fx"]), float(intr["fy"])
    cx, cy = float(intr["cx"]), float(intr["cy"])
    k = (float(intr["k1"]), float(intr["k2"]), float(intr["k3"]), float(intr["k4"]))

    u, v = np.meshgrid(np.arange(width), np.arange(height))
    x_d = (u - cx) / fx
    y_d = (v - cy) / fy
    theta_d = np.sqrt(x_d * x_d + y_d * y_d)
    theta = _solve_theta(theta_d, k)

    # Now (theta, phi) → unit ray. phi from (x_d, y_d) direction.
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    eps = 1e-12
    inv = np.where(theta_d > eps, sin_t / np.maximum(theta_d, eps), 1.0)
    rays = np.stack([x_d * inv, y_d * inv, cos_t], axis=-1)
    # Already unit length by construction (sin^2 + cos^2 = 1).
    return rays.astype(np.float32)


def lift_to_3d(rays: np.ndarray, depth: np.ndarray, depth_convention: str = "z") -> np.ndarray:
    """
    Convert per-pixel rays + depth → 3D camera-frame points (H, W, 3).

    depth_convention:
      * "z"     — depth is distance along the optical (Z) axis
                  (this is what most monocular depth networks output)
      * "ray"   — depth is Euclidean distance along the ray
    """
    if depth_convention == "z":
        # rays[..., 2] is cos(theta); P = depth * ray / ray_z places P at the right Z.
        scale = depth / np.maximum(rays[..., 2], 1e-6)
    elif depth_convention == "ray":
        scale = depth
    else:
        raise ValueError(f"unknown depth_convention: {depth_convention!r}")
    return rays * scale[..., None]


def fit_plane_ransac(points: np.ndarray, threshold: float = 0.10,
                     iterations: int = 2000, rng: np.random.Generator | None = None
                     ) -> tuple[np.ndarray, float, np.ndarray]:
    """
    RANSAC plane fit. `points` is (N, 3). Returns (normal[3], offset, inlier_mask).
    Plane equation: normal · x = offset.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(points)
    best_inliers = np.zeros(n, dtype=bool)
    best_plane = (np.array([0.0, 1.0, 0.0]), 0.0)
    if n < 3:
        return best_plane[0], best_plane[1], best_inliers
    for _ in range(iterations):
        idx = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = points[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        nrm = np.linalg.norm(normal)
        if nrm < 1e-9:
            continue
        normal /= nrm
        offset = float(normal @ p0)
        dist = np.abs(points @ normal - offset)
        inliers = dist < threshold
        if inliers.sum() > best_inliers.sum():
            best_inliers, best_plane = inliers, (normal, offset)
    # Refit on inliers via SVD.
    if best_inliers.sum() >= 3:
        pts = points[best_inliers]
        c = pts.mean(axis=0)
        _, _, vh = np.linalg.svd(pts - c, full_matrices=False)
        normal = vh[-1]
        offset = float(normal @ c)
        best_plane = (normal, offset)
    return best_plane[0], best_plane[1], best_inliers
