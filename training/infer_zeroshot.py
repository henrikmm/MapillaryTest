"""
Zero-shot inference on an unlabelled crop dataset.

Loads all left/right crops from a given root directory and runs the trained
CORAL model, outputting predictions.csv with per-crop class probabilities.

Example:
    python training/infer_zeroshot.py \
        --checkpoint training/runs/20260423-225109/best.pth \
        --crops-root santamarienseZero-shot/raw
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import numpy as np
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from train import build_transforms, predict_from_logits, ID_TO_CLASS


class CropDataset(Dataset):
    def __init__(self, paths: list[tuple[Path, str, str]], image_size: int):
        self.items = paths  # (path, image_id, side)
        self.transform = build_transforms(image_size, train=False)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        path, image_id, side = self.items[index]
        image = np.array(Image.open(path).convert("RGB"))
        image = self.transform(image=image)["image"]
        return {"image": image, "index": index}


def coral_class_probabilities(logits: torch.Tensor) -> torch.Tensor:
    cumulative = torch.sigmoid(logits)
    p_gt_low = cumulative[:, 0]
    p_gt_mid = cumulative[:, 1]
    probs = torch.stack([1.0 - p_gt_low, p_gt_low - p_gt_mid, p_gt_mid], dim=1)
    probs = probs.clamp_min(0.0)
    return probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)


def collect_crops(crops_root: Path) -> list[tuple[Path, str, str]]:
    items: list[tuple[Path, str, str]] = []
    for side in ("left", "right"):
        side_dir = crops_root / "crops" / side
        if not side_dir.exists():
            print(f"[WARN] {side_dir} not found, skipping.")
            continue
        for p in sorted(side_dir.glob("*.jpg")):
            items.append((p, p.stem, side))
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--crops-root", type=Path, required=True,
                        help="Dataset root containing crops/left and crops/right")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output CSV path (default: crops_root/predictions.csv)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_path = args.out or (args.crops_root / "predictions.csv")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    image_size = int(ckpt_args.get("image_size", 384))
    model_name = ckpt_args.get("model", "efficientnet_b2")
    drop_rate = float(ckpt_args.get("drop_rate", 0.3))

    print(f"Model:    {model_name} | image_size={image_size}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Crops root: {args.crops_root}")
    print(f"Device:   {device}")

    items = collect_crops(args.crops_root)
    if not items:
        raise SystemExit("No crops found.")
    print(f"Crops:    {len(items)} ({sum(1 for _,_,s in items if s=='left')} left, "
          f"{sum(1 for _,_,s in items if s=='right')} right)")

    dataset = CropDataset(items, image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = timm.create_model(model_name, pretrained=False, num_classes=2, drop_rate=drop_rate).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    rows: list[dict] = []
    use_amp = device.type == "cuda"

    with torch.no_grad():
        for batch in tqdm(loader, desc="infer", dynamic_ncols=True):
            images = batch["image"].to(device, non_blocking=True)
            indices = batch["index"].tolist()
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(images)
            pred_labels = predict_from_logits(logits).cpu().numpy()
            probs = coral_class_probabilities(logits).cpu().numpy()

            for pos, (idx, pred) in enumerate(zip(indices, pred_labels)):
                path, image_id, side = items[idx]
                class_probs = probs[pos]
                score = float(class_probs[int(pred)])
                ordered = sorted(class_probs.tolist(), reverse=True)
                margin = float(ordered[0] - ordered[1]) if len(ordered) > 1 else 0.0
                rows.append({
                    "image_id":   image_id,
                    "side":       side,
                    "pred_label": ID_TO_CLASS[int(pred)],
                    "pred_id":    int(pred),
                    "score":      round(score, 6),
                    "margin":     round(margin, 6),
                    "prob_baixo": round(float(class_probs[0]), 6),
                    "prob_medio": round(float(class_probs[1]), 6),
                    "prob_alto":  round(float(class_probs[2]), 6),
                })

    rows.sort(key=lambda r: (r["image_id"], r["side"]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    counts = {"Baixo": 0, "Médio": 0, "Alto": 0}
    for r in rows:
        counts[r["pred_label"]] += 1
    total = len(rows)
    print(f"\nPredictions: {total}")
    for label, n in counts.items():
        print(f"  {label}: {n} ({100*n/total:.1f}%)")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
