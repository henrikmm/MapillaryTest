"""
Generate binary masks (terrain only) for all images in images/.
Output: helpers/masks/{image_id}_mask.png — single channel, 255=grass, 0=other.

Terrain-only (label 9): captures ground-level grass precisely.
Vegetation (label 8) was excluded after calibration tests showed it picks up
trees and shrubs, which biased annotation and would confuse the classifier.
"""

import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from tqdm import tqdm

DEFAULT_MODEL_ID = "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"
TARGET_IDS = {9}  # terrain only — vegetation (8) excluded, see docstring


def load_model(device, model_id=DEFAULT_MODEL_ID):
    processor = SegformerImageProcessor.from_pretrained(model_id)
    model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(device).eval()
    return processor, model


def save_mask(args):
    mask, out_path = args
    Image.fromarray(mask).save(out_path)


def run(images_dir, out_dir, batch_size, save_workers, device, model_id=DEFAULT_MODEL_ID):
    out_dir.mkdir(parents=True, exist_ok=True)
    all_images = sorted(images_dir.glob("*.jpg"))
    pending = [p for p in all_images if not (out_dir / (p.stem + "_mask.png")).exists()]

    print(f"Total: {len(all_images)} | Done: {len(all_images)-len(pending)} | Pending: {len(pending)}")
    if not pending:
        print("Nothing to do.")
        return

    processor, model = load_model(device, model_id)

    with ThreadPoolExecutor(max_workers=save_workers) as save_pool:
        with tqdm(total=len(pending), unit="img", dynamic_ncols=True) as bar:
            for i in range(0, len(pending), batch_size):
                batch_paths = pending[i : i + batch_size]
                images = [Image.open(p).convert("RGB") for p in batch_paths]

                inputs = processor(images=images, return_tensors="pt").to(device)
                with torch.no_grad():
                    logits = model(**inputs).logits

                for j, (img, path) in enumerate(zip(images, batch_paths)):
                    w, h = img.size
                    up = F.interpolate(logits[j:j+1], size=(h, w), mode="bilinear", align_corners=False)
                    pred = up.argmax(dim=1).squeeze().cpu().numpy()
                    mask = np.where(np.isin(pred, list(TARGET_IDS)), 255, 0).astype(np.uint8)
                    out_path = out_dir / (path.stem + "_mask.png")
                    save_pool.submit(save_mask, (mask, out_path))

                bar.update(len(batch_paths))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, default=Path("images"))
    parser.add_argument("--out-dir", type=Path, default=Path("helpers/masks"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    args = parser.parse_args()

    print(f"Device: {args.device} | Batch: {args.batch_size} | Model: {args.model}")
    run(args.images_dir, args.out_dir, args.batch_size, args.workers, args.device, args.model)
    print("Done.")


if __name__ == "__main__":
    main()
