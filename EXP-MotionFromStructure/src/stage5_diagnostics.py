"""
Stage 5 — diagnostics.

Reads the artefacts of:
  * stage 0  (sequence selection)
  * stage 1  (intrinsic calibration)
  * stage 2  (per-frame heights)

…and rolls them up into a single report that answers our success criteria
without needing physical ground truth:

  - Did we recover usable intrinsics? (reprojection error, registered frames)
  - Is the recovered camera mount height consistent across frames?
    (cross-frame median + spread of `camera_height_m`)
  - Are terrain (grass) heights ordinally sensible?
    (per-frame p50 distribution; identical scenes should give similar p50s)

Run:
    python -m src.stage5_diagnostics --config config.yaml
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .common import Config, load_config, setup_logging, write_json


def _read(p: Path):
    if not p.exists():
        return None
    try:
        with p.open() as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def build_report(cfg: Config) -> dict:
    log = setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    out = cfg.outputs_dir
    report: dict = {}

    # ---- stage 0 ----
    s0 = _read(out / "stage0_manifest" / "pilot_sequences.json")
    if s0:
        report["stage0"] = {"n_pilot": len(s0["pilot"]), "filters": s0["filters"]}

    # ---- stage 1 (calibration) ----
    s1 = _read(out / "stage1_calib" / "intrinsics.json")
    if s1:
        report["stage1_calibration"] = {
            "ok": s1.get("ok"),
            "n_registered": s1.get("n_registered"),
            "n_input_frames": s1.get("n_input_frames"),
            "mean_reprojection_error_px": s1.get("mean_reprojection_error_px"),
            "intrinsics": {k: s1["intrinsics"].get(k) for k in
                           ("model", "width", "height", "fx", "fy", "cx", "cy")},
        }

    # ---- stage 2 (per-frame heights) ----
    s2_summary = _read(out / "stage2_heights" / "_summary.json")
    if s2_summary:
        ok_frames = [f for f in s2_summary["frames"] if f.get("ok")]
        cross = s2_summary.get("cross_frame_camera_height_m") or {}
        # Cross-frame median of terrain p50 — a coarse "is grass height stable?" signal.
        terrain_p50s = [f.get("terrain_p50_m") for f in ok_frames if f.get("terrain_p50_m") is not None]
        terrain_p50s = [v for v in terrain_p50s if math.isfinite(v)]
        terrain_summary = None
        if terrain_p50s:
            ts = sorted(terrain_p50s)
            terrain_summary = {
                "n_frames_with_terrain": len(ts),
                "median_p50_m": ts[len(ts) // 2],
                "min_p50_m": ts[0],
                "max_p50_m": ts[-1],
            }
        report["stage2_heights"] = {
            "n_processed": len(s2_summary["frames"]),
            "n_ok": len(ok_frames),
            "cross_frame_camera_height": cross,
            "cross_frame_terrain_p50": terrain_summary,
        }

    # ---- success criteria (no GT) ----
    crit: dict = {}
    if "stage1_calibration" in report:
        rep_err = report["stage1_calibration"].get("mean_reprojection_error_px")
        crit["intrinsics_reprojection_below_2px"] = (rep_err is not None and rep_err < 2.0)
    if "stage2_heights" in report:
        cross = report["stage2_heights"]["cross_frame_camera_height"]
        if cross and cross.get("median_m"):
            crit["camera_height_within_30pct_frac_of_frames"] = (
                # frames whose recovered camera-height is within 30% of the cross-frame median
                None  # we only have the aggregate here; detailed per-frame check stays in stage 2 outputs
            )
            crit["camera_height_std_below_30cm"] = (cross.get("std_m", float("inf")) < 0.30)
    report["criteria"] = crit

    out_dir = out / "stage5_diag"
    write_json(out_dir / "report.json", report)
    log.info("wrote %s", out_dir / "report.json")
    if "stage2_heights" in report:
        cross = report["stage2_heights"]["cross_frame_camera_height"]
        if cross:
            log.info("cross-frame camera height: median=%.2fm std=%.2fm n=%d",
                     cross.get("median_m", 0.0), cross.get("std_m", 0.0), cross.get("n", 0))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = load_config(args.config)
    build_report(cfg)


if __name__ == "__main__":
    main()
