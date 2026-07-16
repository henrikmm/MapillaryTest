"""
Stage 3 — scale-align monocular depth to SfM metric scale.

For each frame in each pilot sequence:
    1. Load (or compute) the depth model's per-pixel depth in metres-ish.
    2. Project SfM points visible in this frame to image coordinates and read
       the corresponding model depths at those pixels.
    3. Restrict to road pixels (segmentation).
    4. Compute median(SfM_depth / model_depth) → scale s.
    5. Save scaled depth to outputs/stage3_depth/<seq>/<frame_id>.npz.

Implementation status: depth-loading is wired against the existing
`DINOEXPERIMENT/depth_experiment` outputs if `depth.reuse_existing` is true;
projection of SfM points uses OpenSfM's reconstruction.json. Segmentation gate
is left as a TODO that calls `segmentation.segment` (NotImplementedError until
A6 is resolved).

Run:
    python -m src.stage3_align_depth --config config.yaml [--seq <id>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common import Config, load_config, setup_logging, write_json


def _load_depth_for_frame(cfg: Config, frame_id: str) -> np.ndarray | None:
    """Try to reuse a precomputed depth map from DINOEXPERIMENT; return None if missing."""
    if not cfg["depth"]["reuse_existing"]:
        return None
    dep_dir = cfg.path("depth_experiment_dir")
    # The existing experiment stores per-frame depth as .npz with key "depth"
    # (see DINOEXPERIMENT/depth_experiment/extract_depth.py). Try the conventional path.
    for candidate in [
        dep_dir / "depth" / f"{frame_id}.npz",
        dep_dir / "results" / "depth" / f"{frame_id}.npz",
    ]:
        if candidate.exists():
            with np.load(candidate) as z:
                if "depth" in z.files:
                    return z["depth"].astype(np.float32)
                # fall back to the first array in the archive
                return z[z.files[0]].astype(np.float32)
    return None


def _compute_depth_inline(cfg: Config, image_path: Path) -> np.ndarray:
    """
    Fallback: run the depth model on a single image. Imported lazily so this
    module is usable even without torch.
    """
    from PIL import Image
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    model_id = cfg["depth"]["model_id"]
    device = cfg["depth"]["device"]
    proc = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()
    img = Image.open(image_path).convert("RGB")
    with torch.no_grad():
        inputs = proc(images=img, return_tensors="pt").to(device)
        out = model(**inputs)
        depth = out.predicted_depth[0].cpu().numpy()
    return depth.astype(np.float32)


def align_for_sequence(cfg: Config, seq_id: str) -> dict:
    """
    Scaffold only — for each frame:
      * load depth
      * gather SfM points + project to image (TODO: implement projection per
        OpenSfM camera model — fisheye undistort then perspective).
      * restrict to road via segmentation (TODO once A6 is resolved).
      * compute median ratio, save scaled depth + per-frame report.
    """
    log = setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    recon_path = cfg.outputs_dir / "stage1_sfm" / seq_id / "reconstruction.json"
    manifest = cfg.outputs_dir / "stage0_manifest" / "pilot_sequences.json"
    if not recon_path.exists():
        return {"sequence": seq_id, "ok": False, "error": f"no recon at {recon_path}"}
    with manifest.open() as f:
        pilot = {s["sequence"]: s for s in json.load(f)["pilot"]}
    seq = pilot[seq_id]

    out_dir = cfg.outputs_dir / "stage3_depth" / seq_id
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = cfg.path("images_dir")

    per_frame = []
    for frame in seq["frames"]:
        fid = frame["id"]
        depth = _load_depth_for_frame(cfg, fid)
        used_inline = False
        if depth is None:
            try:
                depth = _compute_depth_inline(cfg, images_dir / f"{fid}.jpg")
                used_inline = True
            except Exception as e:  # noqa: BLE001
                per_frame.append({"frame": fid, "ok": False, "error": f"depth load: {e}"})
                continue

        # TODO: project SfM points into this frame's pixel grid and gather
        # paired (sfm_depth, model_depth) samples — needs a fisheye camera
        # implementation that mirrors OpenSfM's model. Then:
        #   ratios = sfm_depth / model_depth
        #   scale  = np.median(ratios[road_mask])
        # Saving placeholder unscaled depth for now:
        np.savez_compressed(out_dir / f"{fid}.npz", depth=depth, scale=np.float32(1.0))
        per_frame.append({
            "frame": fid, "ok": True, "scale": 1.0, "used_inline_depth": used_inline,
            "todo": "compute per-frame scale once SfM->image projection is implemented",
        })

    summary = {"sequence": seq_id, "n_frames": len(per_frame), "frames": per_frame}
    write_json(out_dir / "_summary.json", summary)
    log.info("seq %s: depth saved for %d frames (scale=1.0 placeholder)", seq_id, len(per_frame))
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
        align_for_sequence(cfg, sid)


if __name__ == "__main__":
    main()
