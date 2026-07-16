"""
Stage 1 — one-shot intrinsic calibration.

The whole dataset uses one physical fisheye camera (see metadata.jsonl —
camera_type=fisheye, width=2704, height=2028). Intrinsics are a rig property,
not a per-frame quantity, so we only need to solve them ONCE.

This stage runs pycolmap on a small contiguous batch of frames (defaults: 25
frames, 1024 px max edge, 2048 SIFT features, single thread) and extracts the
OPENCV_FISHEYE intrinsics from the resulting reconstruction. The numbers get
written to outputs/stage1_calib/intrinsics.json and reused by every downstream
per-frame stage — no per-sequence SfM is required after this.

Run:
    python -m src.stage1_calibrate --config config.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pycolmap

from .common import Config, load_config, setup_logging, write_json


def _select_calibration_frames(cfg: Config, log) -> list[dict]:
    """Pull a short contiguous window of frames from the chosen pilot sub-sequence."""
    cal = cfg["calibrate"]
    pilot_path = cfg.outputs_dir / "stage0_manifest" / "pilot_sequences.json"
    if not pilot_path.exists():
        raise SystemExit(f"missing {pilot_path} — run stage 0 first.")
    with pilot_path.open() as f:
        pilot = json.load(f)["pilot"]
    idx = int(cal.get("source_pilot_index", 0))
    if idx >= len(pilot):
        raise SystemExit(f"source_pilot_index={idx} but only {len(pilot)} pilot sub-sequences")
    seq = pilot[idx]
    start = int(cal.get("start_frame_offset", 0))
    n = int(cal.get("n_frames", 25))
    frames = seq["frames"][start : start + n]
    log.info("calibration: drawing %d frames from sub-sequence %s (offset %d)",
             len(frames), seq["sequence"], start)
    return frames


def _stage_images(frames: list[dict], images_src: Path, project_dir: Path) -> Path:
    img_dir = project_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for f in frames:
        name = f"{f['id']}.jpg"
        src = images_src / name
        dst = img_dir / name
        if not src.exists():
            continue
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)
    return img_dir


def _camera_to_dict(cam: pycolmap.Camera) -> dict:
    """Project a COLMAP Camera to a JSON-friendly dict (OPENCV_FISHEYE schema)."""
    params = list(cam.params)
    out = {
        "model": cam.model.name,
        "width": int(cam.width),
        "height": int(cam.height),
        "params": params,
    }
    if cam.model.name == "OPENCV_FISHEYE" and len(params) == 8:
        fx, fy, cx, cy, k1, k2, k3, k4 = params
        out.update({
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "k1": k1, "k2": k2, "k3": k3, "k4": k4,
            # Sanity-check fields — pixels relative to image dims.
            "fx_norm": fx / max(cam.width, cam.height),
            "principal_point_offset_px": [cx - cam.width / 2.0, cy - cam.height / 2.0],
        })
    return out


def calibrate(cfg: Config) -> dict:
    log = setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    cal = cfg["calibrate"]

    out_dir = cfg.outputs_dir / "stage1_calib"
    out_dir.mkdir(parents=True, exist_ok=True)
    project_dir = out_dir / "colmap_project"
    if project_dir.exists():
        shutil.rmtree(project_dir)
    project_dir.mkdir()

    frames = _select_calibration_frames(cfg, log)
    img_dir = _stage_images(frames, cfg.path("images_dir"), project_dir)
    db_path = project_dir / "database.db"
    sparse_dir = project_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)

    # ---- 1. feature extraction (single shared OPENCV_FISHEYE camera) -----
    reader_opts = pycolmap.ImageReaderOptions()
    reader_opts.camera_model = cal.get("camera_model", "OPENCV_FISHEYE")
    extr_opts = pycolmap.FeatureExtractionOptions()
    extr_opts.max_image_size = int(cal.get("max_image_size", 1024))
    extr_opts.sift.max_num_features = int(cal.get("max_num_features", 2048))
    extr_opts.use_gpu = False  # pip wheel is CPU-only
    log.info("extract: n=%d, max_img=%d, max_feat=%d, threads=%d",
             len(frames), extr_opts.max_image_size,
             extr_opts.sift.max_num_features, int(cal.get("num_threads", 1)))
    pycolmap.extract_features(
        database_path=db_path,
        image_path=img_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader_opts,
        extraction_options=extr_opts,
        device=pycolmap.Device.cpu,
    )

    # ---- 2. sequential matching ------------------------------------------
    log.info("match: sequential overlap=%d", int(cal.get("matching_overlap", 3)))
    match_opts = pycolmap.FeatureMatchingOptions()
    match_opts.use_gpu = False
    pair_opts = pycolmap.SequentialPairingOptions()
    pair_opts.overlap = int(cal.get("matching_overlap", 3))
    pair_opts.quadratic_overlap = False
    pycolmap.match_sequential(
        database_path=db_path,
        matching_options=match_opts,
        pairing_options=pair_opts,
        device=pycolmap.Device.cpu,
    )

    # ---- 3. incremental mapping ------------------------------------------
    log.info("incremental_mapping ...")
    recs = pycolmap.incremental_mapping(
        database_path=db_path,
        image_path=img_dir,
        output_path=sparse_dir,
    )
    if not recs:
        return {"ok": False, "error": "incremental_mapping produced 0 reconstructions"}

    rec_id, rec = max(recs.items(), key=lambda kv: kv[1].num_reg_images())
    log.info("kept reconstruction #%d: %d images, %d points",
             rec_id, rec.num_reg_images(), rec.num_points3D())

    # ---- 4. extract intrinsics ------------------------------------------
    if rec.num_cameras() == 0:
        return {"ok": False, "error": "no camera in reconstruction"}
    cam = next(iter(rec.cameras.values()))
    cam_dict = _camera_to_dict(cam)

    payload = {
        "ok": True,
        "n_input_frames": len(frames),
        "n_registered": rec.num_reg_images(),
        "n_points": rec.num_points3D(),
        "mean_reprojection_error_px": float(rec.compute_mean_reprojection_error()),
        "mean_track_length": float(rec.compute_mean_track_length()),
        "intrinsics": cam_dict,
        "source_pilot_index": int(cal.get("source_pilot_index", 0)),
        "source_frame_ids": [f["id"] for f in frames],
    }
    write_json(out_dir / "intrinsics.json", payload)
    log.info("wrote %s", out_dir / "intrinsics.json")
    if "fx" in cam_dict:
        log.info("  fx=%.1f fy=%.1f cx=%.1f cy=%.1f  (image %dx%d)",
                 cam_dict["fx"], cam_dict["fy"], cam_dict["cx"], cam_dict["cy"],
                 cam_dict["width"], cam_dict["height"])
        log.info("  fx_norm=%.3f  reproj_err=%.2f px",
                 cam_dict["fx_norm"], payload["mean_reprojection_error_px"])
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = load_config(args.config)
    calibrate(cfg)


if __name__ == "__main__":
    main()
