"""
Extrai features exploratórias por crop para análise da altura de grama.

Features extraídas diretamente das máscaras + imagens originais:
  top_ratio        — quão alto no frame a grama chega (proxy de altura real)
  vertical_span    — fração da altura do frame coberta pela grama
  density          — fração de pixels terrain no half-frame
  texture_std      — desvio padrão de luminância na região mascarada (grama alta = mais textura)
  green_saturation — saturação HSV mediana na região (grama verde viva = maior saturação)
  green_dominant   — fração de pixels onde G > R e G > B

Observação:
  Este script NÃO foi usado para anotar nem faz parte do pipeline principal de
  treino. Ele existe apenas como exploração auxiliar.

Output: helpers/exploratory_crop_features.csv
"""

import csv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT        = Path(__file__).parent.parent
MASKS_DIR   = ROOT / "helpers" / "masks"
IMAGES_DIR  = ROOT / "images"
CROPS_DIR   = ROOT / "crops"
OUT_CSV     = Path(__file__).parent / "exploratory_crop_features.csv"
MIN_GRASS   = 5_000


def analyze(args):
    image_id, side = args
    mask_path = MASKS_DIR / f"{image_id}_mask.png"
    img_path  = IMAGES_DIR / f"{image_id}.jpg"
    crop_path = CROPS_DIR / side / f"{image_id}.jpg"

    if not all(p.exists() for p in [mask_path, img_path, crop_path]):
        return None

    mask = np.array(Image.open(mask_path).convert("L"))
    img  = np.array(Image.open(img_path).convert("RGB"))
    h, w = mask.shape
    mid  = w // 2

    col_start = 0   if side == "left" else mid
    col_end   = mid if side == "left" else w

    mask_side = mask[:, col_start:col_end]
    img_side  = img[:, col_start:col_end]

    grass_px = int((mask_side > 0).sum())
    if grass_px < MIN_GRASS:
        return None

    # --- posição vertical da grama no frame ---
    rows     = np.where(np.any(mask_side > 0, axis=1))[0]
    top_row  = int(rows[0])
    bot_row  = int(rows[-1])
    # top_ratio: 1.0 = grama toca o topo do frame (muito alta), 0.0 = só na base
    top_ratio     = round(1.0 - top_row / h, 4)
    vertical_span = round((bot_row - top_row) / h, 4)

    # --- densidade ---
    density = round(grass_px / mask_side.size, 4)

    # --- features de cor/textura nos pixels de grama ---
    px = img_side[mask_side > 0].astype(np.float32)  # (N, 3)

    luminance = 0.299 * px[:, 0] + 0.587 * px[:, 1] + 0.114 * px[:, 2]
    texture_std = round(float(luminance.std()), 2)

    r, g, b = px[:, 0] / 255, px[:, 1] / 255, px[:, 2] / 255
    cmax  = np.maximum(np.maximum(r, g), b)
    cmin  = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin
    sat   = np.where(cmax > 0, delta / cmax, 0.0)
    green_saturation   = round(float(np.median(sat)), 4)
    green_dominant_ratio = round(float(((px[:, 1] > px[:, 0]) & (px[:, 1] > px[:, 2])).mean()), 4)

    return {
        "image_id":            image_id,
        "side":                side,
        "top_ratio":           top_ratio,
        "vertical_span":       vertical_span,
        "density":             density,
        "texture_std":         texture_std,
        "green_saturation":    green_saturation,
        "green_dominant_ratio": green_dominant_ratio,
        "grass_px":            grass_px,
    }


def main():
    tasks = []
    for crop in sorted((CROPS_DIR / "left").glob("*.jpg")):
        tasks.append((crop.stem, "left"))
    for crop in sorted((CROPS_DIR / "right").glob("*.jpg")):
        tasks.append((crop.stem, "right"))

    print(f"Analisando {len(tasks)} crops...")
    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for r in tqdm(pool.map(analyze, tasks), total=len(tasks), unit="crop"):
            if r:
                results.append(r)

    fields = ["image_id", "side", "top_ratio", "vertical_span", "density",
              "texture_std", "green_saturation", "green_dominant_ratio", "grass_px"]

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSalvo: {OUT_CSV} ({len(results)} crops)")
    print_summary(results)


def print_summary(results):
    import statistics
    fields = ["top_ratio", "vertical_span", "density", "texture_std",
              "green_saturation", "green_dominant_ratio"]

    print("\n── Resumo das features ──────────────────────────────────")
    print(f"{'Feature':<22} {'min':>7} {'median':>7} {'max':>7} {'std':>7}")
    print("─" * 58)
    for f in fields:
        vals = [r[f] for r in results]
        print(f"{f:<22} {min(vals):>7.3f} {statistics.median(vals):>7.3f} "
              f"{max(vals):>7.3f} {statistics.stdev(vals):>7.3f}")

    # Score composto simples (normalizado 0-1 depois somado)
    def norm(vals):
        mn, mx = min(vals), max(vals)
        return [(v - mn) / (mx - mn) if mx > mn else 0.5 for v in vals]

    top    = [r["top_ratio"]     for r in results]
    span   = [r["vertical_span"] for r in results]
    dense  = [r["density"]       for r in results]
    tex    = [r["texture_std"]   for r in results]

    scores = [0.4*t + 0.3*s + 0.2*d + 0.1*x
              for t, s, d, x in zip(norm(top), norm(span), norm(dense), norm(tex))]

    for r, sc in zip(results, scores):
        r["score"] = round(sc, 4)

    ranked = sorted(zip(scores, results), reverse=True)

    print("\n── Top 10 provável ALTO ─────────────────────────────────")
    print(f"{'image_id':<20} {'side':<6} {'score':>6} {'top_ratio':>9} {'v_span':>7} {'density':>8}")
    for sc, r in ranked[:10]:
        print(f"{r['image_id']:<20} {r['side']:<6} {sc:>6.3f} "
              f"{r['top_ratio']:>9.3f} {r['vertical_span']:>7.3f} {r['density']:>8.3f}")

    print("\n── Top 10 provável BAIXO ────────────────────────────────")
    print(f"{'image_id':<20} {'side':<6} {'score':>6} {'top_ratio':>9} {'v_span':>7} {'density':>8}")
    for sc, r in ranked[-10:]:
        print(f"{r['image_id']:<20} {r['side']:<6} {sc:>6.3f} "
              f"{r['top_ratio']:>9.3f} {r['vertical_span']:>7.3f} {r['density']:>8.3f}")


if __name__ == "__main__":
    main()
