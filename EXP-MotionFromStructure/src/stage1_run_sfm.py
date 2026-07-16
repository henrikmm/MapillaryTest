"""
Stage 1 — Structure-from-Motion per sub-sequence, using pycolmap (COLMAP 4.x).

Why pycolmap and not OpenSfM:
    OpenSfM is effectively unmaintained and won't build on Python 3.12 without
    significant patching. pycolmap ships Py 3.12 wheels, supports OPENCV_FISHEYE,
    and has a one-call GPS alignment helper (`align_reconstruction_to_locations`).

Pipeline per sub-sequence:
    1. Stage a COLMAP project on disk (images symlinked, fresh database).
    2. Extract SIFT features under the OPENCV_FISHEYE camera model, single camera
       shared across the sub-sequence (intrinsics solved during BA).
    3. Sequential matching — these are vehicle traces, consecutive frames overlap.
    4. Incremental mapping → one or more Reconstructions in COLMAP units.
    5. Pick the largest reconstruction, similarity-align it to GPS-derived ENU
       coordinates so the result is metric. Altitude anchors the vertical axis.
    6. Export to a JSON in the OpenSfM-style schema that stages 2–5 already read:
           {
             cameras: {<camera_id>: {projection_type, width, height, focal, k1, k2}},
             shots:   {<image>: {rotation: [rx,ry,rz], translation: [tx,ty,tz], camera}},
             points:  {<id>: {coordinates: [x,y,z], color: [r,g,b]}}
           }

Run:
    python -m src.stage1_run_sfm --config config.yaml [--limit 3] [--seq <id>]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pycolmap

from .common import (
    Config,
    load_config,
    lonlat_to_local_enu,
    setup_logging,
    write_json,
)


# --------------------------------------------------------------------------- helpers

def _R_to_axis_angle(R: np.ndarray) -> list[float]:
    """3x3 rotation matrix → axis-angle (Rodrigues) vector. cv2-free implementation."""
    cos_t = (np.trace(R) - 1.0) / 2.0
    cos_t = float(np.clip(cos_t, -1.0, 1.0))
    theta = float(np.arccos(cos_t))
    if theta < 1e-9:
        return [0.0, 0.0, 0.0]
    if abs(np.pi - theta) < 1e-6:
        # Numerically stable branch for theta ≈ π.
        diag = np.maximum((np.diag(R) + 1.0) / 2.0, 0.0)
        axis = np.sqrt(diag)
        # Sign disambiguation from off-diagonal entries.
        signs = np.sign([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        axis = axis * np.where(signs == 0, 1.0, signs)
        return (axis * theta).tolist()
    rx = (R[2, 1] - R[1, 2]) / (2 * np.sin(theta))
    ry = (R[0, 2] - R[2, 0]) / (2 * np.sin(theta))
    rz = (R[1, 0] - R[0, 1]) / (2 * np.sin(theta))
    return [float(rx * theta), float(ry * theta), float(rz * theta)]


def _stage_project(seq: dict, images_src: Path, project_dir: Path) -> tuple[Path, dict]:
    """Lay out one COLMAP project on disk; return (image_subdir, name_to_metadata)."""
    img_dir = project_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    name_to_meta: dict[str, dict] = {}
    missing: list[str] = []
    for f in seq["frames"]:
        name = f"{f['id']}.jpg"
        src = images_src / name
        dst = img_dir / name
        if not src.exists():
            missing.append(f["id"])
            continue
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)
        name_to_meta[name] = f

    if missing:
        write_json(project_dir / "missing_images.json", {"missing": missing})
    return img_dir, name_to_meta


def _camera_to_dict(cam: pycolmap.Camera) -> dict:
    """Project a COLMAP Camera onto the OpenSfM-flavoured schema stages 2–5 read."""
    params = list(cam.params)
    # OPENCV_FISHEYE params: [fx, fy, cx, cy, k1, k2, k3, k4]
    if cam.model.name == "OPENCV_FISHEYE" and len(params) == 8:
        fx, fy, cx, cy, k1, k2, k3, k4 = params
        focal = (fx + fy) / 2.0 / max(cam.width, cam.height)  # OpenSfM-style normalised focal
        return {
            "projection_type": "fisheye_opencv",
            "width": int(cam.width), "height": int(cam.height),
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "focal": focal,
            "k1": k1, "k2": k2, "k3": k3, "k4": k4,
            "colmap_model": cam.model.name, "colmap_params": params,
        }
    return {
        "projection_type": cam.model.name.lower(),
        "width": int(cam.width), "height": int(cam.height),
        "colmap_model": cam.model.name, "colmap_params": params,
    }


def _export_reconstruction(rec: pycolmap.Reconstruction) -> dict:
    """Convert a (post-alignment) pycolmap.Reconstruction to a JSON-friendly dict."""
    cameras_out: dict[str, dict] = {}
    for cam_id, cam in rec.cameras.items():
        cameras_out[str(cam_id)] = _camera_to_dict(cam)

    shots_out: dict[str, dict] = {}
    for img_id, img in rec.images.items():
        if not img.has_pose:
            continue
        # cam_from_world: x_cam = R @ x_world + t  — same convention stage 2 expects.
        pose = img.cam_from_world
        R = pose.rotation.matrix()
        t = np.asarray(pose.translation, dtype=float)
        shots_out[img.name] = {
            "rotation": _R_to_axis_angle(R),
            "translation": t.tolist(),
            "camera": str(img.camera_id),
        }

    points_out: dict[str, dict] = {}
    for pid, p in rec.points3D.items():
        points_out[str(pid)] = {
            "coordinates": np.asarray(p.xyz, dtype=float).tolist(),
            "color": np.asarray(p.color, dtype=int).tolist(),
            "error": float(p.error),
        }
    return {"cameras": cameras_out, "shots": shots_out, "points": points_out}


# --------------------------------------------------------------------------- per-sequence driver

def _run_one(seq: dict, images_src: Path, out_root: Path, cfg_sfm: dict, log,
             max_frames: int | None = None) -> dict:
    sid = seq["sequence"]
    seq_out = out_root / sid
    project_dir = seq_out / "colmap_project"
    if project_dir.exists():
        shutil.rmtree(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)

    # Subsample by stride, then optionally cap absolute count (smoke-test mode).
    stride = max(1, int(cfg_sfm.get("frame_stride", 1)))
    frames_in = seq["frames"][::stride]
    if max_frames is not None:
        frames_in = frames_in[:max_frames]
    n_orig = len(seq["frames"])
    seq = {**seq, "frames": frames_in}
    log.info("[%s] using %d/%d frames (stride=%d, max_frames=%s)",
             sid, len(frames_in), n_orig, stride, str(max_frames))

    img_dir, name_to_meta = _stage_project(seq, images_src, project_dir)
    db_path = project_dir / "database.db"
    sparse_dir = project_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)

    device = {
        "auto": pycolmap.Device.auto,
        "cpu":  pycolmap.Device.cpu,
        "cuda": pycolmap.Device.cuda,
    }[str(cfg_sfm.get("device", "auto")).lower()]

    # ---- 1. feature extraction (single shared OPENCV_FISHEYE camera) ----------
    reader_opts = pycolmap.ImageReaderOptions()
    reader_opts.camera_model = cfg_sfm.get("camera_model", "OPENCV_FISHEYE")
    # Build extraction options by mutating defaults; don't touch fields we don't
    # need to override, otherwise COLMAP's internal Check() may reject them.
    extr_opts = pycolmap.FeatureExtractionOptions()
    extr_opts.max_image_size = int(cfg_sfm.get("max_image_size", 1600))
    extr_opts.sift.max_num_features = int(cfg_sfm.get("max_num_features", 4096))
    extr_opts.use_gpu = (device == pycolmap.Device.cuda)
    log.info("[%s] extracting features (n=%d, max_img=%d, max_feat=%d, threads=%d, device=%s)",
             sid, len(name_to_meta),
             extr_opts.max_image_size, extr_opts.sift.max_num_features,
             extr_opts.num_threads, device.name)
    pycolmap.extract_features(
        database_path=db_path,
        image_path=img_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader_opts,
        extraction_options=extr_opts,
        device=device,
    )

    # ---- 2. sequential matching -----------------------------------------------
    log.info("[%s] sequential matching (overlap=%d, device=%s)",
             sid, cfg_sfm.get("matching_overlap", 5), device.name)
    match_opts = pycolmap.FeatureMatchingOptions()
    match_opts.use_gpu = (device == pycolmap.Device.cuda)
    pair_opts = pycolmap.SequentialPairingOptions()
    pair_opts.overlap = int(cfg_sfm.get("matching_overlap", 5))
    pair_opts.quadratic_overlap = False  # keep matching count linear in overlap
    pycolmap.match_sequential(
        database_path=db_path,
        matching_options=match_opts,
        pairing_options=pair_opts,
        device=device,
    )

    # ---- 3. incremental reconstruction ----------------------------------------
    log.info("[%s] incremental mapping", sid)
    recs = pycolmap.incremental_mapping(
        database_path=db_path,
        image_path=img_dir,
        output_path=sparse_dir,
    )
    if not recs:
        return {"sequence": sid, "ok": False, "error": "incremental_mapping produced 0 reconstructions"}

    # Pick the largest reconstruction (most registered images).
    rec_id, rec = max(recs.items(), key=lambda kv: kv[1].num_reg_images())
    log.info("[%s] %d reconstruction(s); kept #%d with %d images, %d points",
             sid, len(recs), rec_id, rec.num_reg_images(), rec.num_points3D())

    # ---- 4. similarity-align to GPS-ENU so the result is metric ---------------
    if not name_to_meta:
        return {"sequence": sid, "ok": False, "error": "no metadata"}
    f0 = next(iter(name_to_meta.values()))
    lon0, lat0, alt0 = f0["lon"], f0["lat"], f0["altitude"]
    tgt_names: list[str] = []
    tgt_locs: list[tuple[float, float, float]] = []
    for img_id in rec.reg_image_ids():
        img = rec.image(img_id)
        meta = name_to_meta.get(img.name)
        if not meta:
            continue
        e, n, u = lonlat_to_local_enu(
            meta["lon"], meta["lat"], meta["altitude"], lon0, lat0, alt0
        )
        tgt_names.append(img.name)
        tgt_locs.append((e, n, u))

    sim3d = pycolmap.align_reconstruction_to_locations(
        rec,
        tgt_image_names=tgt_names,
        tgt_locations=np.asarray(tgt_locs, dtype=float),
        min_common_images=max(3, len(tgt_names) // 4),
        ransac_options=pycolmap.RANSACOptions(),
    )
    if sim3d is None:
        return {"sequence": sid, "ok": False, "error": "GPS alignment (Sim3d) failed"}

    rec.transform(sim3d)
    log.info("[%s] aligned to GPS-ENU: scale=%.4f", sid, float(sim3d.scale))

    # ---- 5. export ------------------------------------------------------------
    payload = _export_reconstruction(rec)
    payload["meta"] = {
        "sequence": sid,
        "n_input_frames": len(seq["frames"]),
        "n_registered": rec.num_reg_images(),
        "n_points": rec.num_points3D(),
        "gps_origin": {"lon": lon0, "lat": lat0, "altitude": alt0},
        "sim3d_scale_colmap_to_metres": float(sim3d.scale),
        "sim3d_translation": np.asarray(sim3d.translation).tolist(),
    }
    write_json(seq_out / "reconstruction.json", payload)

    # Camera centres in metric ENU — useful for diagnostics.
    cam_centres = []
    for img_id in rec.reg_image_ids():
        img = rec.image(img_id)
        if not img.has_pose:
            continue
        pose = img.cam_from_world
        R = pose.rotation.matrix()
        t = np.asarray(pose.translation, dtype=float)
        c = -R.T @ t  # camera centre in world frame (now metric ENU)
        cam_centres.append({"image": img.name, "xyz": c.tolist()})
    write_json(seq_out / "camera_centres_enu.json", {"centres": cam_centres})

    return {
        "sequence": sid,
        "ok": True,
        "n_registered": rec.num_reg_images(),
        "n_input": len(seq["frames"]),
        "n_points": rec.num_points3D(),
        "sim3d_scale": float(sim3d.scale),
    }


# --------------------------------------------------------------------------- entrypoint

def run(cfg: Config, only: list[str] | None = None, limit: int | None = None,
        max_frames: int | None = None) -> None:
    log = setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    manifest_path = cfg.outputs_dir / "stage0_manifest" / "pilot_sequences.json"
    if not manifest_path.exists():
        raise SystemExit(f"missing {manifest_path} — run stage 0 first.")
    with manifest_path.open() as f:
        pilot = json.load(f)["pilot"]

    if only:
        pilot = [s for s in pilot if s["sequence"] in set(only)]
    if limit is not None:
        pilot = pilot[:limit]

    images_src = cfg.path("images_dir")
    out_root = cfg.outputs_dir / "stage1_sfm"
    out_root.mkdir(parents=True, exist_ok=True)

    cfg_sfm = cfg.get("sfm", {}) or {}
    results = []
    for seq in pilot:
        log.info("=== sequence %s (%d frames) ===", seq["sequence"], seq["n_frames"])
        try:
            results.append(_run_one(seq, images_src, out_root, cfg_sfm, log,
                                    max_frames=max_frames))
        except Exception as e:  # noqa: BLE001
            log.exception("[%s] failed", seq["sequence"])
            results.append({"sequence": seq["sequence"], "ok": False, "error": str(e)})
        write_json(out_root / seq["sequence"] / "summary.json", results[-1])

    write_json(out_root / "stage1_summary.json", {"sequences": results})
    n_ok = sum(1 for r in results if r["ok"])
    log.info("stage 1 done: %d/%d sequences reconstructed", n_ok, len(results))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--seq", action="append", default=None,
                    help="restrict to one or more sub-sequence ids (repeatable)")
    ap.add_argument("--limit", type=int, default=None,
                    help="run at most this many sub-sequences from the pilot")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="cap frames per sub-sequence (smoke-test mode)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run(cfg, only=args.seq, limit=args.limit, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
