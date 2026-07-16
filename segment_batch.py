"""
Batch terrain+vegetation isolation for all images in images/.
Saves helper visualizations to helpers/segmented/.

Usage:
    python segment_batch.py
    python segment_batch.py --batch-size 8 --workers 4
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

MODEL_ID = "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"
TARGET_IDS = {8, 9}  # vegetation, terrain


def load_model(device: str):
    processor = SegformerImageProcessor.from_pretrained(MODEL_ID)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID)
    model.to(device).eval()
    return processor, model


TINT = np.array([100, 220, 100], dtype=np.float32)

def save_isolated(args):
    """Runs in a thread — green tint on grass, darken rest, save as JPEG."""
    base_arr, mask, out_path = args
    result = base_arr.astype(np.float32)
    result[mask]  = base_arr[mask]  * 0.5 + TINT * 0.5
    result[~mask] = base_arr[~mask] * 0.6
    Image.fromarray(result.clip(0, 255).astype(np.uint8)).save(out_path, "JPEG", quality=75)


def run(images_dir: Path, out_dir: Path, batch_size: int, save_workers: int, device: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    all_images = sorted(images_dir.glob("*.jpg"))
    pending = [p for p in all_images if not (out_dir / (p.stem + ".jpg")).exists()]

    print(f"Total: {len(all_images)} | Already done: {len(all_images)-len(pending)} | To process: {len(pending)}")
    if not pending:
        print("Nothing to do.")
        return

    processor, model = load_model(device)

    # Pre-load images in background threads while GPU is busy
    def load_image(path: Path):
        return path, Image.open(path).convert("RGB")

    with ThreadPoolExecutor(max_workers=save_workers) as save_pool:
        with tqdm(total=len(pending), unit="img", dynamic_ncols=True) as bar:
            for i in range(0, len(pending), batch_size):
                batch_paths = pending[i : i + batch_size]

                # Load images
                images = [Image.open(p).convert("RGB") for p in batch_paths]
                orig_sizes = [img.size for img in images]  # (W, H)

                # Preprocess — processor handles resizing to model input
                inputs = processor(images=images, return_tensors="pt").to(device)

                with torch.no_grad():
                    logits = model(**inputs).logits  # (B, C, H/4, W/4)

                # Process each image in the batch individually (sizes may vary)
                for j, (img, path, (orig_w, orig_h)) in enumerate(
                    zip(images, batch_paths, orig_sizes)
                ):
                    logit = logits[j : j + 1]
                    up = F.interpolate(logit, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
                    pred = up.argmax(dim=1).squeeze().cpu().numpy()
                    mask = np.isin(pred, list(TARGET_IDS))

                    out_path = out_dir / (path.stem + ".jpg")
                    save_pool.submit(save_isolated, (np.array(img), mask, out_path))

                bar.update(len(batch_paths))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, default=Path("images"))
    parser.add_argument("--out-dir", type=Path, default=Path("helpers/segmented"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4, help="Parallel save threads")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Device: {args.device} | Batch size: {args.batch_size} | Save workers: {args.workers}")
    run(args.images_dir, args.out_dir, args.batch_size, args.workers, args.device)
    print("Done.")


if __name__ == "__main__":
    main()
