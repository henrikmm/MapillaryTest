"""
Undistort the OPENCV_FISHEYE source images (calibrated in stage 1) into
PINHOLE perspective images, so downstream SfM can use the simpler camera
model and so depth-free geometric methods work without fisheye warping.

Inputs:
  * Source fisheye images: <images_dir>/<frame_id>.jpg
  * Fisheye intrinsics:   outputs/stage1_calib/intrinsics.json

Outputs (under EXP-MotionFromStructure/exp_height/):
  * images/<frame_id>.jpg       — undistorted PINHOLE crop
  * intrinsics_pinhole.json     — virtual perspective intrinsics
  * frame_list.json             — list of frame ids that were processed

Notes on FOV choice:
  The source is a 195° fisheye. A perspective projection cannot represent
  more than ~140° HFOV without absurd stretch at the edges, and most SfM
  feature matchers behave best at modest FOVs. We default to 90° HFOV,
  centered, output 1280×960 — that keeps the road and roadside fully visible
  while killing the worst of the radial distortion.

Run:
  python -m exp_height.src.undistort --limit 30
  python -m exp_height.src.undistort --frame-list path.txt
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

EXP_ROOT = Path(__file__).resolve().parents[2]   # EXP-MotionFromStructure/
EH_ROOT = EXP_ROOT / "exp_height"
SRC_INTRINSICS = EXP_ROOT / "outputs" / "stage1_calib" / "intrinsics.json"


def load_fisheye_intrinsics(path: Path):
    with path.open() as f:
        d = json.load(f)["intrinsics"]
    K = np.array([[d["fx"], 0,       d["cx"]],
                  [0,       d["fy"], d["cy"]],
                  [0,       0,       1]], dtype=np.float64)
    D = np.array([[d["k1"]], [d["k2"]], [d["k3"]], [d["k4"]]], dtype=np.float64)
    return K, D, int(d["width"]), int(d["height"])


def make_pinhole_K(out_w: int, out_h: int, hfov_deg: float):
    fx = out_w / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    fy = fx  # square pixels in the virtual camera
    K_new = np.array([[fx, 0,  out_w / 2.0],
                      [0,  fy, out_h / 2.0],
                      [0,  0,  1.0]], dtype=np.float64)
    return K_new, fx, fy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", type=Path,
                    default=Path("/home/henri/projects/MapillaryTest/santamarienseZero-shot/raw/images"))
    ap.add_argument("--out-w", type=int, default=1280)
    ap.add_argument("--out-h", type=int, default=960)
    ap.add_argument("--hfov-deg", type=float, default=90.0)
    ap.add_argument("--limit", type=int, default=None,
                    help="process at most N frames from the source pilot[0] sub-sequence")
    ap.add_argument("--frame-list", type=Path, default=None,
                    help="explicit newline-separated list of frame ids to process")
    ap.add_argument("--quality", type=int, default=92)
    args = ap.parse_args()

    K, D, src_w, src_h = load_fisheye_intrinsics(SRC_INTRINSICS)
    K_new, fx, fy = make_pinhole_K(args.out_w, args.out_h, args.hfov_deg)
    print(f"source fisheye: {src_w}x{src_h}  fx={K[0,0]:.1f} fy={K[1,1]:.1f}")
    print(f"target pinhole: {args.out_w}x{args.out_h}  HFOV={args.hfov_deg:.1f}°  fx={fx:.1f}")

    # Precompute the remap once — shared across frames.
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K_new,
        (args.out_w, args.out_h), cv2.CV_16SC2,
    )

    # Frame selection
    if args.frame_list:
        frame_ids = [ln.strip() for ln in args.frame_list.read_text().splitlines() if ln.strip()]
    else:
        manifest = EXP_ROOT / "outputs" / "stage0_manifest" / "pilot_sequences.json"
        with manifest.open() as f:
            seq = json.load(f)["pilot"][0]
        frame_ids = [f["id"] for f in seq["frames"]]
    if args.limit is not None:
        frame_ids = frame_ids[: args.limit]

    out_dir = EH_ROOT / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    done = []
    for fid in frame_ids:
        src = args.images_dir / f"{fid}.jpg"
        if not src.exists():
            print(f"  miss {fid}")
            continue
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None or img.shape[:2] != (src_h, src_w):
            print(f"  skip {fid} (shape {img.shape if img is not None else None})")
            continue
        und = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT)
        cv2.imwrite(str(out_dir / f"{fid}.jpg"), und,
                    [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
        done.append(fid)

    intr_out = {
        "model": "PINHOLE",
        "width": args.out_w, "height": args.out_h,
        "fx": fx, "fy": fy,
        "cx": args.out_w / 2.0, "cy": args.out_h / 2.0,
        "hfov_deg": args.hfov_deg,
        "source_intrinsics": str(SRC_INTRINSICS),
    }
    (EH_ROOT / "intrinsics_pinhole.json").write_text(json.dumps(intr_out, indent=2))
    (EH_ROOT / "frame_list.json").write_text(json.dumps({"n": len(done), "frames": done}, indent=2))
    print(f"wrote {len(done)} undistorted images to {out_dir}")
    print(f"wrote intrinsics to {EH_ROOT / 'intrinsics_pinhole.json'}")


if __name__ == "__main__":
    main()
