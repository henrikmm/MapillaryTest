"""
Stage 4 — per-pixel vegetation height (cm above the metric ground plane).

For each frame in each sequence:
    For every pixel p classified as `vegetation`:
        ray r = back-projected pixel (camera-fisheye model)
        canopy point P_c = camera_center + r * scaled_depth(p)
        ground point P_g = intersect(ray r, ground plane)
        height(p) = (P_c - P_g) · plane_normal      # signed distance, m

Outputs (per frame):
    outputs/stage4_height/<seq>/<frame_id>.npz       (height map, valid mask)
    outputs/stage4_height/<seq>/<frame_id>.json      (patch-level summary stats)

This is a scaffold — the geometry is straightforward but depends on a working
fisheye unprojector matching OpenSfM's camera model. That helper is sketched in
`_unproject_fisheye_pixel` below and should be validated against OpenSfM's
`Camera.pixel_bearing` before trusting any numbers.

Run:
    python -m src.stage4_vegetation_height --config config.yaml [--seq <id>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common import Config, load_config, setup_logging, write_json


def _unproject_fisheye_pixel(uv: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """
    Back-project pixel coords (N, 2) to unit bearing vectors in camera frame.
    `K` is the intrinsic matrix, `dist` = [k1, k2] for OpenSfM's fisheye model.
    Validate against OpenSfM's `Camera.pixel_bearing` before relying on this.
    """
    raise NotImplementedError(
        "Fisheye unprojection must mirror OpenSfM's camera model. "
        "Implement after stage 1 produces a real reconstruction.json so we can "
        "diff this function against OpenSfM's `Camera.pixel_bearing` on real samples."
    )


def _ray_plane_intersection(origin: np.ndarray, direction: np.ndarray,
                            normal: np.ndarray, offset: float) -> np.ndarray:
    """Intersect rays (origin + t * direction) with plane normal·x = offset. Returns t (N,)."""
    denom = direction @ normal
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (offset - origin @ normal) / denom
    t[~np.isfinite(t)] = np.nan
    return t


def heights_for_frame(cfg: Config, seq_id: str, frame_id: str,
                      plane_normal: np.ndarray, plane_offset: float) -> dict:
    """
    Skeleton for the per-frame computation. Returns a dict; writes nothing until
    the unproject + segmentation TODOs are filled in.
    """
    return {
        "frame": frame_id,
        "ok": False,
        "todo": [
            "implement _unproject_fisheye_pixel against OpenSfM's Camera.pixel_bearing",
            "wire src.segmentation.segment(image, ['vegetation']) for the canopy mask",
            "load scaled depth from stage 3 outputs (stage3_depth/<seq>/<frame>.npz)",
            "compute height = (canopy_point - ground_point) · plane_normal, in metres",
            "write outputs/stage4_height/<seq>/<frame>.npz + .json summaries",
        ],
    }


def run_for_sequence(cfg: Config, seq_id: str) -> dict:
    log = setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    gp_path = cfg.outputs_dir / "stage2_ground" / seq_id / "ground_plane.json"
    if not gp_path.exists():
        return {"sequence": seq_id, "ok": False, "error": f"missing {gp_path}"}
    with gp_path.open() as f:
        gp = json.load(f)
    if not gp.get("ok"):
        return {"sequence": seq_id, "ok": False, "error": "stage 2 did not pass"}

    normal = np.array(gp["plane"]["normal"], dtype=float)
    offset = float(gp["plane"]["offset"])

    manifest = cfg.outputs_dir / "stage0_manifest" / "pilot_sequences.json"
    with manifest.open() as f:
        pilot = {s["sequence"]: s for s in json.load(f)["pilot"]}
    seq = pilot[seq_id]

    per_frame = [heights_for_frame(cfg, seq_id, f["id"], normal, offset) for f in seq["frames"]]
    summary = {"sequence": seq_id, "n_frames": len(per_frame), "frames": per_frame}
    out_dir = cfg.outputs_dir / "stage4_height" / seq_id
    write_json(out_dir / "_summary.json", summary)
    log.info("seq %s: stage 4 scaffolded for %d frames (no heights computed yet)",
             seq_id, len(per_frame))
    return summary


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
        run_for_sequence(cfg, sid)


if __name__ == "__main__":
    main()
