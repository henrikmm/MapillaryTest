#!/usr/bin/env python3
import json
import math
import os
import time
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


DEFAULT_SEQUENCE_ID = "by41kl5i2vQmCTx0NOsqfV"
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
ENV_PATH = PROJECT_ROOT / ".env"
API_BASE = "https://graph.mapillary.com"
RESOLUTION_FIELDS = {
    "256": "thumb_256_url",
    "1024": "thumb_1024_url",
    "2048": "thumb_2048_url",
    "original": "thumb_original_url",
}

IMAGE_FIELDS = [
    "id",
    "altitude",
    "atomic_scale",
    "camera_parameters",
    "camera_type",
    "captured_at",
    "compass_angle",
    "computed_altitude",
    "computed_compass_angle",
    "computed_geometry",
    "computed_rotation",
    "creator",
    "exif_orientation",
    "geometry",
    "height",
    "make",
    "model",
    "thumb_256_url",
    "thumb_1024_url",
    "thumb_2048_url",
    "thumb_original_url",
    "merge_cc",
    "mesh",
    "quality_score",
    "sequence",
    "sfm_cluster",
    "width",
]


def load_env(path: Path) -> dict[str, str]:
    values = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def session_from_env() -> requests.Session:
    env = load_env(ENV_PATH)
    token = os.getenv("MAPILLARY_ACCESS_TOKEN") or env.get("MAPILLARY_ACCESS_TOKEN")
    api_base = os.getenv("MAPILLARY_API_BASE") or env.get("MAPILLARY_API_BASE")
    if api_base:
        global API_BASE
        API_BASE = api_base.rstrip("/")
    if not token:
        raise SystemExit("MAPILLARY_ACCESS_TOKEN nao encontrado no ambiente nem em .env")
    s = requests.Session()
    s.headers["Authorization"] = f"OAuth {token}"
    return s


def get_json(session: requests.Session, url: str, params: dict | None = None) -> dict:
    for attempt in range(3):
        response = session.get(url, params=params, timeout=30)
        if response.ok:
            return response.json()
        if attempt == 2:
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            raise RuntimeError(f"{response.status_code} {response.url}: {detail}")
        time.sleep(2**attempt)
    raise AssertionError("unreachable")


def fetch_sequence_ids(session: requests.Session, sequence_id: str) -> list[str]:
    ids = []
    url = f"{API_BASE}/image_ids"
    params = {"sequence_id": sequence_id}
    while url:
        data = get_json(session, url, params=params)
        ids.extend(item["id"] for item in data.get("data", []))
        url = data.get("paging", {}).get("next")
        params = None
    return ids


def probe_fields(session: requests.Session, image_id: str) -> dict:
    result = {}
    for field in IMAGE_FIELDS:
        url = f"{API_BASE}/{image_id}"
        try:
            data = get_json(session, url, params={"fields": field})
            result[field] = {
                "ok": True,
                "present": field in data,
                "example": data.get(field),
            }
        except Exception as exc:
            result[field] = {"ok": False, "error": str(exc)}
    return result


def fetch_image_metadata(session: requests.Session, image_id: str, fields: list[str]) -> dict:
    return get_json(
        session,
        f"{API_BASE}/{image_id}",
        params={"fields": ",".join(fields)},
    )


def fov_from_camera_parameters(meta: dict) -> dict | None:
    params = meta.get("camera_parameters")
    width = meta.get("width")
    height = meta.get("height")
    if not params or len(params) < 1 or not width or not height:
        return None

    focal = params[0]
    max_dim = max(width, height)
    fx_px = focal * max_dim
    fy_px = focal * max_dim
    if fx_px <= 0 or fy_px <= 0:
        return None

    return {
        "focal_normalized": focal,
        "k1": params[1] if len(params) > 1 else None,
        "k2": params[2] if len(params) > 2 else None,
        "fx_px_assuming_opensfm_norm": fx_px,
        "fy_px_assuming_opensfm_norm": fy_px,
        "horizontal_fov_deg_assuming_pinhole": math.degrees(2 * math.atan(width / (2 * fx_px))),
        "vertical_fov_deg_assuming_pinhole": math.degrees(2 * math.atan(height / (2 * fy_px))),
        "diagonal_fov_deg_assuming_pinhole": math.degrees(
            2 * math.atan(math.hypot(width, height) / (2 * fx_px))
        ),
    }


def summarize(records: list[dict], field_probe: dict, sequence_id: str) -> dict:
    field_presence = {}
    for field in IMAGE_FIELDS:
        present = sum(1 for rec in records if field in rec and rec[field] is not None)
        field_presence[field] = {
            "probe_ok": field_probe.get(field, {}).get("ok", False),
            "present_count": present,
            "total_records": len(records),
        }

    camera_types = Counter(rec.get("camera_type") for rec in records)
    makes = Counter(rec.get("make") for rec in records if rec.get("make"))
    models = Counter(rec.get("model") for rec in records if rec.get("model"))
    sizes = Counter(f"{rec.get('width')}x{rec.get('height')}" for rec in records)

    fovs = [rec["_derived_fov"] for rec in records if rec.get("_derived_fov")]
    fov_stats = {}
    if fovs:
        for key in [
            "horizontal_fov_deg_assuming_pinhole",
            "vertical_fov_deg_assuming_pinhole",
            "diagonal_fov_deg_assuming_pinhole",
            "focal_normalized",
            "k1",
            "k2",
        ]:
            values = [item[key] for item in fovs if item.get(key) is not None]
            if values:
                fov_stats[key] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                }

    examples_by_camera = defaultdict(list)
    for rec in records:
        ctype = rec.get("camera_type") or "missing"
        if len(examples_by_camera[ctype]) < 3:
            examples_by_camera[ctype].append(
                {
                    "id": rec.get("id"),
                    "width": rec.get("width"),
                    "height": rec.get("height"),
                    "camera_parameters": rec.get("camera_parameters"),
                    "derived_fov": rec.get("_derived_fov"),
                }
            )

    return {
        "sequence_id": sequence_id,
        "image_count": len(records),
        "field_presence": field_presence,
        "camera_types": dict(camera_types),
        "makes": dict(makes),
        "models": dict(models),
        "image_sizes": dict(sizes),
        "fov_stats_assuming_opensfm_normalized_pinhole": fov_stats,
        "examples_by_camera_type": dict(examples_by_camera),
    }


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def download_image(session: requests.Session, url: str, dest: Path) -> bool:
    if dest.exists():
        return False
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    for attempt in range(3):
        try:
            response = session.get(url, timeout=60, stream=True)
            response.raise_for_status()
            with tmp.open("wb") as out:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        out.write(chunk)
            tmp.replace(dest)
            return True
        except Exception:
            tmp.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    return False


def download_images(
    session: requests.Session,
    records: list[dict],
    images_dir: Path,
    resolution: str,
    workers: int,
) -> dict:
    images_dir.mkdir(parents=True, exist_ok=True)
    field = RESOLUTION_FIELDS[resolution]
    jobs = []
    for rec in records:
        url = rec.get(field)
        if not url:
            continue
        jobs.append((url, images_dir / f"{rec['id']}.jpg"))

    downloaded = 0
    skipped = 0
    failed = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_image, session, url, dest): dest
            for url, dest in jobs
        }
        for idx, future in enumerate(as_completed(futures), start=1):
            dest = futures[future]
            try:
                if future.result():
                    downloaded += 1
                else:
                    skipped += 1
            except Exception as exc:
                failed.append({"path": str(dest), "error": str(exc)})
            if idx % 100 == 0 or idx == len(jobs):
                print(f"images {idx}/{len(jobs)}")

    return {
        "resolution": resolution,
        "field": field,
        "requested": len(jobs),
        "downloaded": downloaded,
        "skipped_existing": skipped,
        "failed": failed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default=DEFAULT_SEQUENCE_ID)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--download-images", action="store_true")
    parser.add_argument("--resolution", choices=sorted(RESOLUTION_FIELDS), default="original")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    session = session_from_env()
    ids = fetch_sequence_ids(session, args.sequence)
    if not ids:
        raise SystemExit(f"Nenhuma imagem encontrada para a sequencia {args.sequence}")

    write_json(out_dir / "sequence_image_ids.json", {"sequence_id": args.sequence, "ids": ids})

    field_probe = probe_fields(session, ids[0])
    write_json(out_dir / "field_probe_first_image.json", field_probe)

    valid_fields = [field for field, info in field_probe.items() if info.get("ok")]
    records = []
    with (out_dir / "metadata_all.jsonl").open("w") as out:
        for idx, image_id in enumerate(ids, start=1):
            meta = fetch_image_metadata(session, image_id, valid_fields)
            meta["_derived_fov"] = fov_from_camera_parameters(meta)
            records.append(meta)
            out.write(json.dumps(meta, ensure_ascii=False) + "\n")
            if idx % 100 == 0 or idx == len(ids):
                print(f"{idx}/{len(ids)}")

    write_json(out_dir / "sample_metadata_first_image.json", records[0])
    summary = summarize(records, field_probe, args.sequence)
    if args.download_images:
        summary["download"] = download_images(
            session,
            records,
            out_dir / "images",
            args.resolution,
            args.workers,
        )
    write_json(out_dir / "summary.json", summary)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
