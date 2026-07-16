"""
SfM on the undistorted PINHOLE images, GPS-anchored for metric scale.

Inputs:
  exp_height/images/*.jpg                 — produced by undistort.py
  exp_height/intrinsics_pinhole.json      — analytic PINHOLE intrinsics
  exp_height/frame_list.json              — ordered frame ids
  santamarienseZero-shot/raw/metadata.jsonl — for GPS lon/lat/altitude

Outputs (under exp_height/sfm/):
  colmap_project/                          — pycolmap workspace (db, sparse/)
  reconstruction.json                      — frames, poses, points (in metres)
  alignment.json                           — Sim3d transform + GPS reference

Run:
  python -m exp_height.src.sfm
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import pycolmap

EXP_ROOT = Path(__file__).resolve().parents[2]
EH_ROOT = EXP_ROOT / "exp_height"
import os
META = Path(os.environ.get("MFS_METADATA",
            "/home/henri/projects/MapillaryTest/metadata.jsonl"))

_EARTH_R = 6_378_137.0


def lonlat_to_local_enu(lon, lat, alt, lon0, lat0, alt0):
    lat0r = math.radians(lat0)
    east = math.radians(lon - lon0) * _EARTH_R * math.cos(lat0r)
    north = math.radians(lat - lat0) * _EARTH_R
    up = alt - alt0
    return east, north, up


def load_gps_index() -> dict[str, tuple[float, float, float]]:
    """frame_id -> (lon, lat, alt)"""
    out = {}
    with META.open() as f:
        for ln in f:
            d = json.loads(ln)
            geom = d.get("geometry") or {}
            coords = geom.get("coordinates")
            if not coords:
                continue
            out[str(d["id"])] = (float(coords[0]), float(coords[1]), float(d.get("altitude", 0.0)))
    return out


def stage_images(frame_ids: list[str], img_src: Path, project_dir: Path) -> Path:
    img_dir = project_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for fid in frame_ids:
        src = img_src / f"{fid}.jpg"
        dst = img_dir / f"{fid}.jpg"
        if not src.exists():
            continue
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)
    return img_dir


def run(args):
    out_dir = EH_ROOT / "sfm"
    project_dir = out_dir / "colmap_project"
    if project_dir.exists():
        shutil.rmtree(project_dir)
    project_dir.mkdir(parents=True)

    intr = json.loads((EH_ROOT / "intrinsics_pinhole.json").read_text())
    frames = json.loads((EH_ROOT / "frame_list.json").read_text())["frames"]
    print(f"sfm: {len(frames)} frames, intrinsics={intr['model']} {intr['width']}x{intr['height']} fx={intr['fx']:.1f}")

    img_dir = stage_images(frames, EH_ROOT / "images", project_dir)
    db_path = project_dir / "database.db"
    sparse_dir = project_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)

    # ---- 1. extract features with FIXED PINHOLE intrinsics --------------
    reader_opts = pycolmap.ImageReaderOptions()
    reader_opts.camera_model = "PINHOLE"
    reader_opts.camera_params = f"{intr['fx']},{intr['fy']},{intr['cx']},{intr['cy']}"
    extr_opts = pycolmap.FeatureExtractionOptions()
    extr_opts.max_image_size = int(args.max_image_size)
    extr_opts.sift.max_num_features = int(args.max_num_features)
    extr_opts.use_gpu = False
    print(f"extract: max_img={extr_opts.max_image_size} max_feat={extr_opts.sift.max_num_features}")
    pycolmap.extract_features(
        database_path=db_path,
        image_path=img_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader_opts,
        extraction_options=extr_opts,
        device=pycolmap.Device.cpu,
    )

    # ---- 2. sequential matching -----------------------------------------
    print(f"match: sequential overlap={args.matching_overlap}")
    match_opts = pycolmap.FeatureMatchingOptions()
    match_opts.use_gpu = False
    pair_opts = pycolmap.SequentialPairingOptions()
    pair_opts.overlap = int(args.matching_overlap)
    pair_opts.quadratic_overlap = False
    pycolmap.match_sequential(
        database_path=db_path,
        matching_options=match_opts,
        pairing_options=pair_opts,
        device=pycolmap.Device.cpu,
    )

    # ---- 3. incremental mapping -----------------------------------------
    print("incremental_mapping ...")
    recs = pycolmap.incremental_mapping(
        database_path=db_path,
        image_path=img_dir,
        output_path=sparse_dir,
    )
    if not recs:
        raise SystemExit("incremental_mapping produced 0 reconstructions")

    rec_id, rec = max(recs.items(), key=lambda kv: kv[1].num_reg_images())
    print(f"reconstruction #{rec_id}: {rec.num_reg_images()} images, {rec.num_points3D()} points, "
          f"reproj_err={rec.compute_mean_reprojection_error():.2f} px")

    # ---- 4. GPS alignment -----------------------------------------------
    gps_idx = load_gps_index()
    image_names = []
    locations = []
    for img in rec.images.values():
        fid = Path(img.name).stem
        if fid in gps_idx:
            image_names.append(img.name)
            locations.append(gps_idx[fid])
    if len(locations) < 3:
        raise SystemExit(f"only {len(locations)} GPS-known images registered — cannot align")

    lon0, lat0, alt0 = locations[0]
    enu = np.array([lonlat_to_local_enu(lo, la, al, lon0, lat0, alt0)
                    for (lo, la, al) in locations], dtype=np.float64)

    ransac = pycolmap.RANSACOptions()
    ransac.max_error = 5.0       # 5 m RANSAC threshold for GPS noise
    sim3 = pycolmap.align_reconstruction_to_locations(
        rec, image_names, enu, 3, ransac,
    )
    if sim3 is None:
        raise SystemExit("GPS alignment failed (no Sim3d)")
    rec.transform(sim3)
    print(f"aligned to ENU: scale={sim3.scale:.4f} (post-warp; meters)")

    # ---- 5. export ------------------------------------------------------
    images_out = []
    for img in rec.images.values():
        if not img.has_pose:
            continue
        # World-from-camera: invert cam-from-world stored in image.cam_from_world
        cfw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
        R_cw = np.array(cfw.rotation.matrix())
        t_cw = np.array(cfw.translation)
        Rwc = R_cw.T
        twc = -Rwc @ t_cw
        images_out.append({
            "name": img.name,
            "frame_id": Path(img.name).stem,
            "R_wc": Rwc.tolist(),
            "t_wc": twc.tolist(),
        })

    points_out = []
    for pt_id, pt in rec.points3D.items():
        points_out.append({
            "id": int(pt_id),
            "xyz": pt.xyz.tolist(),
            "color": [int(c) for c in pt.color],
            "error": float(pt.error),
            "track": [(el.image_id, el.point2D_idx) for el in pt.track.elements],
        })

    cam = next(iter(rec.cameras.values()))
    write_dir = out_dir
    write_dir.mkdir(exist_ok=True)
    (write_dir / "reconstruction.json").write_text(json.dumps({
        "ok": True,
        "n_input_frames": len(frames),
        "n_registered": rec.num_reg_images(),
        "n_points": rec.num_points3D(),
        "mean_reprojection_error_px": float(rec.compute_mean_reprojection_error()),
        "intrinsics_used": intr,
        "intrinsics_post_sfm": {
            "model": cam.model.name, "width": cam.width, "height": cam.height,
            "params": list(cam.params),
        },
        "images": images_out,
        "points": points_out,
    }, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else o))

    (write_dir / "alignment.json").write_text(json.dumps({
        "gps_origin": {"lon": lon0, "lat": lat0, "alt": alt0},
        "sim3_scale": float(sim3.scale),
        "n_aligned_cameras": len(image_names),
        "ransac_threshold_m": 5.0,
    }, indent=2))

    # quick consistency: distances between consecutive registered cameras in metres
    cams_by_name = {im["name"]: np.array(im["t_wc"]) for im in images_out}
    inter = []
    prev = None
    for fid in frames:
        nm = f"{fid}.jpg"
        if nm in cams_by_name:
            cur = cams_by_name[nm]
            if prev is not None:
                inter.append(float(np.linalg.norm(cur - prev)))
            prev = cur
    if inter:
        a = np.array(inter)
        print(f"inter-frame camera step (m): n={len(a)} median={np.median(a):.2f} "
              f"min={a.min():.2f} max={a.max():.2f}")

    print(f"wrote {write_dir / 'reconstruction.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-image-size", type=int, default=1280)
    ap.add_argument("--max-num-features", type=int, default=4096)
    ap.add_argument("--matching-overlap", type=int, default=5)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
