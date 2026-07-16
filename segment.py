"""
Semantic segmentation using SegFormer-B1 (Cityscapes 1024x1024).
Outputs a color-coded overlay with only the target classes highlighted.

Usage:
    python segment.py images/1933479480908877.jpg
    python segment.py images/1933479480908877.jpg --out result.png
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

MODEL_ID = "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"

# Cityscapes label index → name
CITYSCAPES_LABELS = {
    0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence",
    5: "pole", 6: "traffic light", 7: "traffic sign", 8: "vegetation",
    9: "terrain", 10: "sky", 11: "person", 12: "rider", 13: "car",
    14: "truck", 15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle",
}

TARGET_CLASSES = {"road", "traffic sign", "vegetation", "terrain", "sky", "car", "truck", "bus"}

# Distinct colors per target class (RGB)
CLASS_COLORS = {
    "road":         (128, 64,  128),
    "traffic sign": (220, 220,  0),
    "vegetation":   ( 107, 142, 35),
    "terrain":      (152, 251, 152),
    "sky":          ( 70, 130, 180),
    "car":          (  0,   0, 142),
    "truck":        (  0,   0,  70),
    "bus":          (  0,  60, 100),
}


def load_model(device: str):
    processor = SegformerImageProcessor.from_pretrained(MODEL_ID)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID)
    model.to(device).eval()
    return processor, model


def segment(image_path: Path, processor, model, device: str) -> dict:
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits  # (1, num_classes, H/4, W/4)

    # Upsample to original size
    upsampled = torch.nn.functional.interpolate(
        logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False
    )
    pred = upsampled.argmax(dim=1).squeeze().cpu().numpy()  # (H, W)

    # Build RGBA overlay — only target classes get color, rest transparent
    overlay = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
    class_pixels = {}
    total_pixels = orig_h * orig_w

    for label_id, label_name in CITYSCAPES_LABELS.items():
        if label_name not in TARGET_CLASSES:
            continue
        mask = pred == label_id
        pixel_count = int(mask.sum())
        if pixel_count == 0:
            continue
        class_pixels[label_name] = pixel_count
        color = CLASS_COLORS[label_name]
        overlay[mask] = (*color, 180)  # semi-transparent

    # Blend overlay onto original (60% opacity on masked regions)
    base = np.array(image)
    alpha = overlay[..., 3:4] / 255.0
    blended = (base * (1 - alpha * 0.55) + overlay[..., :3] * alpha * 0.55).astype(np.uint8)
    result = Image.fromarray(blended).convert("RGBA")

    # Legend rendered directly on top-left corner of the image
    draw = ImageDraw.Draw(result)
    pad = 12
    row_h = 28
    legend_classes = sorted(class_pixels.items(), key=lambda x: -x[1])
    box_w = 230
    box_h = pad + row_h * len(legend_classes) + pad

    # Semi-transparent dark background for legend
    legend_bg = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 170))
    result.paste(legend_bg, (pad, pad), legend_bg)

    draw = ImageDraw.Draw(result)
    y = pad * 2
    for name, count in legend_classes:
        pct = count / total_pixels * 100
        color = CLASS_COLORS[name]
        # Color swatch
        draw.rectangle([pad * 2, y, pad * 2 + 20, y + 18], fill=(*color, 255))
        # Label + percentage
        draw.text((pad * 2 + 28, y), f"{name:<14} {pct:5.1f}%", fill=(255, 255, 255, 255))
        y += row_h

    return {"image": result.convert("RGB"), "class_pixels": class_pixels, "total_pixels": total_pixels}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print("Loading model (first run downloads ~200 MB)...")
    processor, model = load_model(args.device)

    print(f"Segmenting {args.input}...")
    result = segment(args.input, processor, model, args.device)

    out_path = args.out or Path("helpers/segmented") / (args.input.stem + "_segmented.png")
    result["image"].save(out_path)

    print(f"\nSaved → {out_path}")
    print("\nClass coverage:")
    total = result["total_pixels"]
    for name, count in sorted(result["class_pixels"].items(), key=lambda x: -x[1]):
        print(f"  {name:<15} {count/total*100:5.1f}%  ({count:,} px)")


if __name__ == "__main__":
    main()
