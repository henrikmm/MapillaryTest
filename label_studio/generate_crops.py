"""
Para cada imagem, gera dois crops (esquerdo e direito):
- Pixels de grama: original em qualidade máxima
- Resto: preto
- Recortado no bounding box da grama daquele lado
- Salvo em crops/left/ e crops/right/

Pula lados onde não há grama suficiente (< MIN_GRASS_PX pixels).
"""

import argparse
import json
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from tqdm import tqdm

IMAGE_SERVER = "http://localhost:8081"
MIN_GRASS_PX = 5_000   # ignora lados com grama demais pequena
PADDING      = 20      # px de contexto extra ao redor do bbox

# Paths — set by main() after arg parsing
IMAGES_DIR = MASKS_DIR = LEFT_DIR = RIGHT_DIR = TASKS_OUT = None


def crop_side(img_arr, mask_arr, col_start, col_end, out_path, image_id, side):
    """Gera um crop de um lado. Retorna True se gerado, False se sem grama."""
    mask_side = mask_arr[:, col_start:col_end]
    if mask_side.sum() // 255 < MIN_GRASS_PX:
        return False

    img_side = img_arr[:, col_start:col_end].copy()

    # Zera tudo que não é grama
    img_side[mask_side == 0] = 0

    # Bounding box da grama
    rows = np.any(mask_side > 0, axis=1)
    cols = np.any(mask_side > 0, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]

    # Adiciona padding
    h, w = mask_side.shape
    r0 = max(0, r0 - PADDING)
    r1 = min(h - 1, r1 + PADDING)
    c0 = max(0, c0 - PADDING)
    c1 = min(w - 1, c1 + PADDING)

    crop = img_side[r0:r1, c0:c1]
    Image.fromarray(crop).save(out_path, "JPEG", quality=95)
    return True


def process(img_path):
    image_id = img_path.stem
    mask_path = MASKS_DIR / f"{image_id}_mask.png"
    if not mask_path.exists():
        return []

    img_arr  = np.array(Image.open(img_path).convert("RGB"))
    mask_arr = np.array(Image.open(mask_path).convert("L"))
    h, w     = mask_arr.shape
    mid      = w // 2

    tasks = []
    for side, col_start, col_end, out_dir in [
        ("left",  0,   mid, LEFT_DIR),
        ("right", mid, w,   RIGHT_DIR),
    ]:
        out_path = out_dir / f"{image_id}.jpg"
        if out_path.exists():
            tasks.append((image_id, side, out_path))
            continue
        ok = crop_side(img_arr, mask_arr, col_start, col_end, out_path, image_id, side)
        if ok:
            tasks.append((image_id, side, out_path))

    return tasks


def main():
    global IMAGES_DIR, MASKS_DIR, LEFT_DIR, RIGHT_DIR, TASKS_OUT

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent.parent,
                        help="Dataset root (default: repo root)")
    args = parser.parse_args()

    root       = args.root
    IMAGES_DIR = root / "images"
    MASKS_DIR  = root / "helpers" / "masks"
    LEFT_DIR   = root / "crops" / "left"
    RIGHT_DIR  = root / "crops" / "right"
    TASKS_OUT  = root / "label_studio" / "tasks.json"

    LEFT_DIR.mkdir(parents=True, exist_ok=True)
    RIGHT_DIR.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(IMAGES_DIR.glob("*.jpg"))
    all_tasks = []

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(tqdm(pool.map(process, img_paths), total=len(img_paths), unit="img"))

    for result in results:
        for image_id, side, out_path in result:
            all_tasks.append({
                "data": {
                    "image_id": image_id,
                    "side":     side,
                    "original": f"{IMAGE_SERVER}/crops/{side}/{image_id}.jpg",
                }
            })

    random.seed(42)
    random.shuffle(all_tasks)

    TASKS_OUT.parent.mkdir(parents=True, exist_ok=True)
    TASKS_OUT.write_text(json.dumps(all_tasks, indent=2))
    unique_images = len({task["data"]["image_id"] for task in all_tasks})
    print(f"\nGerados: {len(all_tasks)} tasks de {unique_images} image_ids")
    print(f"Tasks → {TASKS_OUT}")


if __name__ == "__main__":
    main()
