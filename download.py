"""
Download a Mapillary sequence: images + metadata.

Usage:
    python download.py                  # uses .env defaults
    python download.py --resolution 1024
    python download.py --workers 8
"""

import os
import json
import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.getenv("MAPILLARY_API_BASE", "https://graph.mapillary.com")
ACCESS_TOKEN = os.getenv("MAPILLARY_ACCESS_TOKEN")
SEQUENCE_KEY = os.getenv("MAPILLARY_SEQUENCE_KEY")

METADATA_FIELDS = (
    "id,sequence,captured_at,geometry,compass_angle,is_pano,"
    "width,height,altitude,camera_type,creator,"
    "thumb_256_url,thumb_1024_url,thumb_2048_url,thumb_original_url"
)

RESOLUTION_FIELD = {
    "256": "thumb_256_url",
    "1024": "thumb_1024_url",
    "2048": "thumb_2048_url",
    "original": "thumb_original_url",
}


def auth_session() -> requests.Session:
    s = requests.Session()
    s.headers["Authorization"] = f"OAuth {ACCESS_TOKEN}"
    return s


def fetch_sequence_ids(session: requests.Session, sequence_id: str) -> list[str]:
    ids = []
    url = f"{API_BASE}/image_ids?sequence_id={sequence_id}"
    while url:
        r = session.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        ids.extend(d["id"] for d in data["data"])
        url = data.get("paging", {}).get("next")
    return ids


def fetch_metadata(session: requests.Session, image_id: str) -> dict:
    url = f"{API_BASE}/{image_id}?fields={METADATA_FIELDS}"
    for attempt in range(3):
        try:
            r = session.get(url, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def download_image(session: requests.Session, url: str, dest: Path) -> bool:
    """Returns True if downloaded, False if skipped (already exists)."""
    if dest.exists():
        return False
    tmp = dest.with_suffix(".tmp")
    for attempt in range(3):
        try:
            r = session.get(url, timeout=60, stream=True)
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(65536):
                    f.write(chunk)
            tmp.rename(dest)
            return True
        except requests.RequestException:
            tmp.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def process_image(
    image_id: str,
    session: requests.Session,
    images_dir: Path,
    resolution_field: str,
) -> dict | None:
    meta = fetch_metadata(session, image_id)
    img_url = meta.get(resolution_field)
    if img_url:
        dest = images_dir / f"{image_id}.jpg"
        download_image(session, img_url, dest)
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default=SEQUENCE_KEY)
    parser.add_argument("--resolution", default="original",
                        choices=["256", "1024", "2048", "original"])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    if not ACCESS_TOKEN:
        raise SystemExit("MAPILLARY_ACCESS_TOKEN not set")
    if not args.sequence:
        raise SystemExit("MAPILLARY_SEQUENCE_KEY not set")

    out = Path(args.output_dir)
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out / "metadata.jsonl"

    res_field = RESOLUTION_FIELD[args.resolution]

    # Load already-processed IDs to support resume
    done_ids: set[str] = set()
    if metadata_path.exists():
        with open(metadata_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass

    session = auth_session()

    print(f"Fetching image IDs for sequence {args.sequence}...")
    all_ids = fetch_sequence_ids(session, args.sequence)
    remaining = [i for i in all_ids if i not in done_ids]
    print(f"Total: {len(all_ids)} | Already done: {len(done_ids)} | To fetch: {len(remaining)}")

    if not remaining:
        print("Nothing to do.")
        return

    with open(metadata_path, "a") as meta_file:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_image, img_id, session, images_dir, res_field): img_id
                for img_id in remaining
            }
            with tqdm(total=len(remaining), unit="img") as bar:
                for future in as_completed(futures):
                    img_id = futures[future]
                    try:
                        meta = future.result()
                        meta_file.write(json.dumps(meta) + "\n")
                        meta_file.flush()
                    except Exception as e:
                        tqdm.write(f"[ERROR] {img_id}: {e}")
                    bar.update(1)

    print(f"\nDone. Images: {images_dir}  Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
