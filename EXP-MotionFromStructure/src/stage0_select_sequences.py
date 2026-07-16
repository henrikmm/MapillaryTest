"""
Stage 0 — sequence selection.

Reads metadata.jsonl, groups by `sequence`, drops panoramas, and emits a JSON
manifest of the sequences that pass the SfM-readiness filters defined in
config.yaml -> stage0_select.

Output: outputs/stage0_manifest/pilot_sequences.json
        outputs/stage0_manifest/all_sequences_summary.json

Run:
    python -m src.stage0_select_sequences --config config.yaml
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from .common import (
    Config,
    haversine_m,
    load_config,
    read_jsonl,
    setup_logging,
    write_json,
)


def _frame_record(rec: dict) -> dict | None:
    """Project a raw metadata record down to the fields stage 1 needs."""
    try:
        lon, lat = rec["geometry"]["coordinates"]
    except (KeyError, TypeError, ValueError):
        return None
    return {
        "id": rec["id"],
        "sequence": rec.get("sequence"),
        "captured_at": rec.get("captured_at"),
        "lon": float(lon),
        "lat": float(lat),
        "altitude": float(rec.get("altitude", 0.0) or 0.0),
        "compass_angle": rec.get("compass_angle"),
        "is_pano": bool(rec.get("is_pano", False)),
        "width": rec.get("width"),
        "height": rec.get("height"),
        "camera_type": rec.get("camera_type"),
    }


def _summarise_sequence(seq_id: str, frames: list[dict]) -> dict:
    """Compute path length, gap stats, and a healthy-track flag."""
    frames = sorted(frames, key=lambda f: f["captured_at"] or 0)
    gaps_m: list[float] = []
    total_m = 0.0
    for a, b in zip(frames, frames[1:]):
        d = haversine_m(a["lon"], a["lat"], b["lon"], b["lat"])
        gaps_m.append(d)
        total_m += d
    return {
        "sequence": seq_id,
        "n_frames": len(frames),
        "track_length_m": round(total_m, 2),
        "max_gap_m": round(max(gaps_m), 2) if gaps_m else 0.0,
        "median_gap_m": round(sorted(gaps_m)[len(gaps_m) // 2], 2) if gaps_m else 0.0,
        "altitude_min": round(min(f["altitude"] for f in frames), 2),
        "altitude_max": round(max(f["altitude"] for f in frames), 2),
        "altitude_range_m": round(
            max(f["altitude"] for f in frames) - min(f["altitude"] for f in frames), 2
        ),
        "any_pano": any(f["is_pano"] for f in frames),
        "camera_types": sorted({f["camera_type"] for f in frames if f["camera_type"]}),
        "frames": frames,
    }


def _split_by_size(chunks: list[list[dict]], max_n: int, overlap: int) -> list[list[dict]]:
    """Cap each chunk at `max_n` frames, with `overlap` shared frames between adjacent windows."""
    if max_n <= 0:
        return chunks
    out: list[list[dict]] = []
    step = max(1, max_n - max(0, overlap))
    for c in chunks:
        if len(c) <= max_n:
            out.append(c)
            continue
        i = 0
        while i < len(c):
            out.append(c[i : i + max_n])
            if i + max_n >= len(c):
                break
            i += step
    return out


def _split_on_gaps(seq_id: str, frames: list[dict], split_gap_m: float) -> list[tuple[str, list[dict]]]:
    """
    Mapillary often packs a whole capture session into a single sequence id.
    Where the GPS jump between consecutive frames exceeds `split_gap_m`, treat
    that as a session break and start a new sub-sequence id `<seq>__<idx>`.
    """
    frames = sorted(frames, key=lambda f: f["captured_at"] or 0)
    chunks: list[list[dict]] = [[]]
    for i, f in enumerate(frames):
        if not chunks[-1]:
            chunks[-1].append(f)
            continue
        prev = chunks[-1][-1]
        gap = haversine_m(prev["lon"], prev["lat"], f["lon"], f["lat"])
        if gap > split_gap_m:
            chunks.append([f])
        else:
            chunks[-1].append(f)
    return [(f"{seq_id}__{i:03d}", c) for i, c in enumerate(chunks) if c]


def _windowed(seq_id: str, frames: list[dict], split_gap_m: float,
              max_n: int, overlap: int) -> list[tuple[str, list[dict]]]:
    """Gap-split, then size-cap with overlap. Renumbers chunks at the end."""
    gap_chunks = [c for _, c in _split_on_gaps(seq_id, frames, split_gap_m)]
    sized = _split_by_size(gap_chunks, max_n, overlap)
    return [(f"{seq_id}__{i:03d}", c) for i, c in enumerate(sized) if c]


def select(cfg: Config) -> dict:
    log = setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    sel = cfg["stage0_select"]
    metadata_path: Path = cfg.path("metadata_jsonl")

    by_seq: dict[str, list[dict]] = defaultdict(list)
    n_total = n_pano = n_skipped = 0
    for rec in read_jsonl(metadata_path):
        n_total += 1
        f = _frame_record(rec)
        if f is None:
            n_skipped += 1
            continue
        if sel["exclude_panoramas"] and f["is_pano"]:
            n_pano += 1
            continue
        if f["sequence"] is None:
            n_skipped += 1
            continue
        by_seq[f["sequence"]].append(f)

    log.info(
        "read %d records: %d panoramas dropped, %d skipped, %d sequences",
        n_total, n_pano, n_skipped, len(by_seq),
    )

    # Split each Mapillary sequence on session-break-sized GPS gaps, then cap chunk size.
    sub_sequences: list[tuple[str, list[dict]]] = []
    for sid, fs in by_seq.items():
        sub_sequences.extend(_windowed(
            sid, fs,
            split_gap_m=sel["split_gap_m"],
            max_n=sel["max_frames_per_subsequence"],
            overlap=sel["subsequence_overlap"],
        ))
    log.info(
        "split into %d sub-sequences (split_gap=%.1fm, max_frames=%d, overlap=%d)",
        len(sub_sequences), sel["split_gap_m"],
        sel["max_frames_per_subsequence"], sel["subsequence_overlap"],
    )

    summaries = [_summarise_sequence(sid, fs) for sid, fs in sub_sequences]

    def passes(s: dict) -> tuple[bool, str]:
        if s["n_frames"] < sel["min_frames_per_sequence"]:
            return False, f"too few frames ({s['n_frames']})"
        if s["track_length_m"] < sel["min_track_length_m"]:
            return False, f"track too short ({s['track_length_m']:.1f}m)"
        return True, "ok"

    decorated = []
    for s in summaries:
        ok, reason = passes(s)
        decorated.append({**s, "passes": ok, "reason": reason})

    eligible = [s for s in decorated if s["passes"]]
    # Pilot ranking: prefer mid-length sequences (not the noisiest, not the smallest).
    eligible.sort(key=lambda s: (-s["n_frames"], s["max_gap_m"]))
    pilot = eligible[: sel["pilot_size"]]

    log.info(
        "%d/%d sequences pass filters; %d picked for pilot",
        len(eligible), len(decorated), len(pilot),
    )

    out_dir = cfg.outputs_dir / "stage0_manifest"
    out_dir.mkdir(parents=True, exist_ok=True)

    # The "all sequences" file omits per-frame lists to stay compact.
    summary_compact = [{k: v for k, v in s.items() if k != "frames"} for s in decorated]
    write_json(out_dir / "all_sequences_summary.json", {
        "n_total_records": n_total,
        "n_panoramas_dropped": n_pano,
        "n_skipped": n_skipped,
        "sequences": summary_compact,
        "filters": sel,
    })

    pilot_payload = {
        "filters": sel,
        "pilot": pilot,  # full per-frame info — stage 1 consumes this
    }
    write_json(out_dir / "pilot_sequences.json", pilot_payload)

    log.info("wrote %s", out_dir / "pilot_sequences.json")
    return pilot_payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = load_config(args.config)
    select(cfg)


if __name__ == "__main__":
    main()
