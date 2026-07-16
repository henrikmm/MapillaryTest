"""
Export model predictions back to Label Studio.

Outputs:
  - label_studio_predictions.json: payload for Label Studio import predictions API
  - review_queue.csv: disagreements and uncertain cases sorted by review priority
  - predictions.csv: all predictions with human labels and model scores

Example:
    python training/export_label_studio_predictions.py \
        --checkpoint training/runs/20260423-190255/best.pth
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import numpy as np
import requests
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from train import CLASS_TO_ID, ID_TO_CLASS, build_transforms, predict_from_logits


ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class PredictionSample:
    task_id: int
    image_id: str
    side: str
    path: Path
    human_label: int
    human_label_name: str
    human_confidence: str


class PredictionDataset(Dataset):
    def __init__(self, samples: list[PredictionSample], image_size: int):
        self.samples = samples
        self.transform = build_transforms(image_size, train=False)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        image = np.array(Image.open(sample.path).convert("RGB"))
        image = self.transform(image=image)["image"]
        return {
            "image": image,
            "index": index,
        }


def load_samples(root: Path, annotation_path: Path) -> list[PredictionSample]:
    data = json.loads(annotation_path.read_text())
    samples: list[PredictionSample] = []
    skipped = 0

    for item in data:
        task_id = item.get("id")
        task_data = item.get("data") or {}
        image_id = task_data.get("image_id")
        side = task_data.get("side")
        annotations = item.get("annotations") or []

        if not task_id or not image_id or side not in {"left", "right"} or not annotations:
            skipped += 1
            continue

        label_name = None
        confidence = None
        for result in annotations[0].get("result") or []:
            choices = (result.get("value") or {}).get("choices") or []
            if not choices:
                continue
            if result.get("from_name") == "height_class":
                label_name = choices[0]
            elif result.get("from_name") == "confidence":
                confidence = choices[0]

        if label_name not in CLASS_TO_ID:
            skipped += 1
            continue

        crop_path = root / "crops" / side / f"{image_id}.jpg"
        if not crop_path.exists():
            skipped += 1
            continue

        samples.append(
            PredictionSample(
                task_id=int(task_id),
                image_id=image_id,
                side=side,
                path=crop_path,
                human_label=CLASS_TO_ID[label_name],
                human_label_name=label_name,
                human_confidence=confidence or "",
            )
        )

    if skipped:
        print(f"[WARN] Ignored {skipped} items without valid task/crop/annotation.")
    return samples


def coral_class_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Approximate class probabilities from CORAL cumulative probabilities."""
    cumulative = torch.sigmoid(logits)
    p_gt_low = cumulative[:, 0]
    p_gt_mid = cumulative[:, 1]
    probs = torch.stack(
        [
            1.0 - p_gt_low,
            p_gt_low - p_gt_mid,
            p_gt_mid,
        ],
        dim=1,
    )
    probs = probs.clamp_min(0.0)
    return probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)


def review_priority(abs_error: int, score: float, margin: float, human_confidence: str) -> int:
    if abs_error == 2:
        return 1
    if abs_error == 1 and score >= 0.75:
        return 2
    if abs_error == 1:
        return 3
    if human_confidence == "certo" and score <= 0.55:
        return 4
    if human_confidence == "incerto" and score >= 0.85:
        return 5
    if margin <= 0.10:
        return 6
    return 9


def priority_reason(priority: int) -> str:
    return {
        1: "discordancia_de_2_classes",
        2: "discordancia_alta_confianca",
        3: "discordancia",
        4: "humano_certo_modelo_baixa_confianca",
        5: "humano_incerto_modelo_alta_confianca",
        6: "modelo_baixa_margem",
        9: "baixa_prioridade",
    }[priority]


def mislabeling_score(priority: int, abs_error: int, score: float, margin: float) -> float:
    if priority == 1:
        return 1.0
    if priority == 2:
        return 0.90
    if priority == 3:
        return 0.70
    if priority in {4, 5, 6}:
        return 0.40 + min(0.30, (1.0 - margin) * 0.30)
    return max(0.0, min(0.20, abs_error * 0.10 + (1.0 - score) * 0.10))


def prediction_result(label_name: str) -> list[dict]:
    return [
        {
            "from_name": "height_class",
            "to_name": "original",
            "type": "choices",
            "value": {"choices": [label_name]},
        }
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def upload_predictions(
    label_studio_url: str,
    project_id: int,
    api_token: str,
    predictions: list[dict],
    batch_size: int,
) -> None:
    base = label_studio_url.rstrip("/")
    url = f"{base}/api/projects/{project_id}/import/predictions"
    headers = {
        "Authorization": f"Token {api_token}",
        "Content-Type": "application/json",
    }
    created = 0
    for start in range(0, len(predictions), batch_size):
        chunk = predictions[start : start + batch_size]
        response = requests.post(url, headers=headers, json=chunk, timeout=60)
        response.raise_for_status()
        payload = response.json()
        created += int(payload.get("created") or 0)
        print(f"[upload] sent={start + len(chunk)}/{len(predictions)} created={created}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--annotations", type=Path, default=ROOT / "MapillaryDatasetL+R-Annoted.json"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--model-version", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--label-studio-url", default=os.getenv("LABEL_STUDIO_URL", "http://localhost:8080"))
    parser.add_argument("--project-id", type=int, default=os.getenv("LABEL_STUDIO_PROJECT_ID"))
    parser.add_argument("--api-token", default=os.getenv("LABEL_STUDIO_API_TOKEN"))
    parser.add_argument("--upload-batch-size", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get("args", {})

    image_size = int(ckpt_args.get("image_size", 384))
    model_name = ckpt_args.get("model", "efficientnet_b2")
    drop_rate = float(ckpt_args.get("drop_rate", 0.3))
    out_dir = args.out_dir or args.checkpoint.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    model_version = args.model_version or f"{model_name}_coral_{image_size}_{args.checkpoint.parent.name}"

    samples = load_samples(args.root, args.annotations)
    dataset = PredictionDataset(samples, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=2,
        drop_rate=drop_rate,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    predictions_payload: list[dict] = []
    all_rows: list[dict] = []

    use_amp = device.type == "cuda"
    with torch.no_grad():
        for batch in tqdm(loader, desc="predict", dynamic_ncols=True):
            images = batch["image"].to(device, non_blocking=True)
            indices = batch["index"].tolist()
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(images)

            pred_labels = predict_from_logits(logits).cpu().numpy()
            probs = coral_class_probabilities(logits).cpu().numpy()

            for batch_pos, (row_idx, pred_label) in enumerate(zip(indices, pred_labels)):
                sample = samples[row_idx]
                class_probs = probs[batch_pos]
                score = float(class_probs[int(pred_label)])
                ordered = sorted(class_probs.tolist(), reverse=True)
                margin = float(ordered[0] - ordered[1]) if len(ordered) > 1 else 0.0
                pred_name = ID_TO_CLASS[int(pred_label)]
                abs_error = abs(sample.human_label - int(pred_label))
                priority = review_priority(
                    abs_error=abs_error,
                    score=score,
                    margin=margin,
                    human_confidence=sample.human_confidence,
                )
                mislabeling = mislabeling_score(priority, abs_error, score, margin)

                predictions_payload.append(
                    {
                        "task": sample.task_id,
                        "model_version": model_version,
                        "score": score,
                        "cluster": priority,
                        "mislabeling": mislabeling,
                        "result": prediction_result(pred_name),
                    }
                )

                all_rows.append(
                    {
                        "review_priority": priority,
                        "review_reason": priority_reason(priority),
                        "task_id": sample.task_id,
                        "image_id": sample.image_id,
                        "side": sample.side,
                        "path": str(sample.path.relative_to(args.root)),
                        "human_label": sample.human_label_name,
                        "human_confidence": sample.human_confidence,
                        "pred_label": pred_name,
                        "abs_error": abs_error,
                        "score": round(score, 6),
                        "margin": round(margin, 6),
                        "mislabeling": round(mislabeling, 6),
                        "prob_baixo": round(float(class_probs[0]), 6),
                        "prob_medio": round(float(class_probs[1]), 6),
                        "prob_alto": round(float(class_probs[2]), 6),
                    }
                )

    predictions_path = out_dir / "label_studio_predictions.json"
    predictions_path.write_text(json.dumps(predictions_payload, indent=2, ensure_ascii=False))

    all_rows.sort(
        key=lambda row: (
            row["review_priority"],
            -row["abs_error"],
            -row["score"],
            row["task_id"],
        )
    )
    write_csv(out_dir / "predictions.csv", all_rows)
    write_csv(
        out_dir / "review_queue.csv",
        [row for row in all_rows if row["review_priority"] < 9],
    )

    disagreements = sum(1 for row in all_rows if row["abs_error"] > 0)
    severe = sum(1 for row in all_rows if row["abs_error"] == 2)
    print(f"\nSaved: {predictions_path}")
    print(f"Saved: {out_dir / 'predictions.csv'}")
    print(f"Saved: {out_dir / 'review_queue.csv'}")
    print(f"Predictions: {len(predictions_payload)} | disagreements: {disagreements} | severe: {severe}")
    print(f"Model version: {model_version}")

    if args.upload:
        if not args.project_id:
            raise SystemExit("--project-id or LABEL_STUDIO_PROJECT_ID is required with --upload")
        if not args.api_token:
            raise SystemExit("--api-token or LABEL_STUDIO_API_TOKEN is required with --upload")
        upload_predictions(
            label_studio_url=args.label_studio_url,
            project_id=int(args.project_id),
            api_token=args.api_token,
            predictions=predictions_payload,
            batch_size=args.upload_batch_size,
        )


if __name__ == "__main__":
    main()
