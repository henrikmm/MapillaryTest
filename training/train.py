"""
Train an ordinal grass-height classifier from the annotated crop dataset.

Baseline:
  - EfficientNet-B2 from timm
  - CORAL ordinal head with 2 outputs for 3 classes
  - 384x384 letterbox preprocessing
  - grouped split by image_id
  - confidence-weighted loss: certo=1.0, incerto=0.6
  - RTX 4060 Ti 8 GB friendly defaults: batch=8, grad accumulation=2, AMP on

Example:
    python training/train.py
    python training/train.py --dry-run --no-pretrained --device cpu --num-workers 0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from contextlib import nullcontext
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2
import numpy as np
import timm
import torch
from albumentations.pytorch import ToTensorV2
from coral_pytorch.dataset import levels_from_labelbatch, proba_to_label
from coral_pytorch.losses import CoralLoss
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent
CLASS_TO_ID = {"Baixo": 0, "Médio": 1, "Alto": 2}
ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}
CONFIDENCE_WEIGHT = {"certo": 1.0, "incerto": 0.6}


@dataclass(frozen=True)
class Sample:
    image_id: str
    side: str
    path: Path
    label: int
    confidence: str
    weight: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_annotations(root: Path, annotation_path: Path) -> list[Sample]:
    data = json.loads(annotation_path.read_text())
    samples: list[Sample] = []
    skipped = 0

    for item in data:
        task_data = item.get("data") or {}
        image_id = task_data.get("image_id")
        side = task_data.get("side")
        annotations = item.get("annotations") or []
        if not image_id or side not in {"left", "right"} or not annotations:
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

        confidence = confidence if confidence in CONFIDENCE_WEIGHT else "incerto"
        crop_path = root / "crops" / side / f"{image_id}.jpg"
        if not crop_path.exists():
            skipped += 1
            continue

        samples.append(
            Sample(
                image_id=image_id,
                side=side,
                path=crop_path,
                label=CLASS_TO_ID[label_name],
                confidence=confidence,
                weight=CONFIDENCE_WEIGHT[confidence],
            )
        )

    if skipped:
        print(f"[WARN] Ignored {skipped} malformed or missing annotation items.")
    return samples


def group_label(labels: list[int]) -> int:
    """One stratification label per image_id. Preserve rare high-grass groups."""
    return max(labels)


def split_samples(
    samples: list[Sample],
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    total = train_size + val_size + test_size
    train_size, val_size, test_size = (
        train_size / total,
        val_size / total,
        test_size / total,
    )

    labels_by_group: dict[str, list[int]] = defaultdict(list)
    for sample in samples:
        labels_by_group[sample.image_id].append(sample.label)

    groups = np.array(sorted(labels_by_group))
    y_group = np.array([group_label(labels_by_group[g]) for g in groups])

    first_split = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=seed
    )
    train_val_idx, test_idx = next(first_split.split(groups, y_group))
    train_val_groups = groups[train_val_idx]
    train_val_y = y_group[train_val_idx]
    test_groups = set(groups[test_idx])

    val_relative_size = val_size / (train_size + val_size)
    second_split = StratifiedShuffleSplit(
        n_splits=1, test_size=val_relative_size, random_state=seed + 1
    )
    train_idx, val_idx = next(second_split.split(train_val_groups, train_val_y))
    train_groups = set(train_val_groups[train_idx])
    val_groups = set(train_val_groups[val_idx])

    train, val, test = [], [], []
    for sample in samples:
        if sample.image_id in train_groups:
            train.append(sample)
        elif sample.image_id in val_groups:
            val.append(sample)
        elif sample.image_id in test_groups:
            test.append(sample)
        else:
            raise RuntimeError(f"Sample group was not assigned: {sample.image_id}")

    return train, val, test


def print_split_summary(name: str, samples: list[Sample]) -> None:
    labels = Counter(s.label for s in samples)
    conf = Counter(s.confidence for s in samples)
    groups = {s.image_id for s in samples}
    label_text = ", ".join(f"{ID_TO_CLASS[i]}={labels[i]}" for i in range(3))
    conf_text = ", ".join(f"{k}={v}" for k, v in sorted(conf.items()))
    print(
        f"{name:<5} samples={len(samples):4d} groups={len(groups):4d} | "
        f"{label_text} | {conf_text}"
    )


def build_transforms(image_size: int, train: bool, hflip_p: float = 0.0) -> A.Compose:
    transforms = [
        A.LongestMaxSize(max_size=image_size, interpolation=cv2.INTER_CUBIC),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=(0, 0, 0),
            position="center",
        ),
    ]
    if train:
        if hflip_p > 0:
            transforms.append(A.HorizontalFlip(p=hflip_p))
        transforms.extend(
            [
                A.ColorJitter(
                    brightness=0.20,
                    contrast=0.20,
                    saturation=0.15,
                    hue=0.02,
                    p=0.70,
                ),
                A.GaussNoise(p=0.15),
                A.GaussianBlur(blur_limit=(3, 5), p=0.15),
            ]
        )
    transforms.extend(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    return A.Compose(transforms)


class GrassCropDataset(Dataset):
    def __init__(self, samples: list[Sample], transform: A.Compose):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        image = np.array(Image.open(sample.path).convert("RGB"))
        image = self.transform(image=image)["image"]
        return {
            "image": image,
            "label": torch.tensor(sample.label, dtype=torch.long),
            "weight": torch.tensor(sample.weight, dtype=torch.float32),
            "confidence": sample.confidence,
            "image_id": sample.image_id,
            "side": sample.side,
        }


def make_loader(
    samples: list[Sample],
    transform: A.Compose,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        GrassCropDataset(samples, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        drop_last=False,
    )


def set_requires_grad(module: torch.nn.Module, value: bool) -> None:
    for param in module.parameters():
        param.requires_grad = value


def freeze_for_head_training(model: torch.nn.Module) -> None:
    set_requires_grad(model, False)
    classifier = model.get_classifier()
    if isinstance(classifier, torch.nn.Module):
        set_requires_grad(classifier, True)
    else:
        for name, param in model.named_parameters():
            if "classifier" in name or "fc" in name:
                param.requires_grad = True


def unfreeze_last_blocks(model: torch.nn.Module, n_blocks: int) -> None:
    set_requires_grad(model, False)

    for module_name in ("classifier", "conv_head", "bn2", "global_pool"):
        module = getattr(model, module_name, None)
        if isinstance(module, torch.nn.Module):
            set_requires_grad(module, True)

    blocks = getattr(model, "blocks", None)
    if isinstance(blocks, torch.nn.Sequential):
        for block in list(blocks.children())[-n_blocks:]:
            set_requires_grad(block, True)
    else:
        # Generic fallback: unfreeze the last 25% of parameters by module order.
        params = list(model.parameters())
        start = int(len(params) * 0.75)
        for param in params[start:]:
            param.requires_grad = True


def trainable_parameter_count(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def make_optimizer(
    model: torch.nn.Module,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name or "fc" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr})
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def weighted_coral_loss(
    criterion: CoralLoss,
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    levels = levels_from_labelbatch(labels, num_classes=3).to(logits.device)
    per_sample = criterion(logits, levels)
    weights = sample_weights.to(logits.device)
    return (per_sample * weights).sum() / weights.sum().clamp_min(1e-6)


def predict_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return proba_to_label(torch.sigmoid(logits)).long()


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: CoralLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    grad_accum_steps: int,
    amp: bool,
    desc: str,
) -> tuple[float, np.ndarray, np.ndarray, list[str]]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_items = 0
    all_true: list[int] = []
    all_pred: list[int] = []
    all_conf: list[str] = []

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc=desc, dynamic_ncols=True)
    for step, batch in enumerate(pbar, start=1):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        weights = batch["weight"].to(device, non_blocking=True)
        autocast_context = (
            torch.amp.autocast(device_type="cuda", enabled=True)
            if amp
            else nullcontext()
        )

        with torch.set_grad_enabled(is_train):
            with autocast_context:
                logits = model(images)
                loss = weighted_coral_loss(criterion, logits, labels, weights)

            if is_train:
                scaled_loss = loss / grad_accum_steps
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                if step % grad_accum_steps == 0 or step == len(loader):
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        batch_size = labels.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_items += batch_size

        preds = predict_from_logits(logits.detach()).cpu().numpy()
        all_true.extend(labels.detach().cpu().numpy().tolist())
        all_pred.extend(preds.tolist())
        all_conf.extend(batch["confidence"])
        pbar.set_postfix(loss=total_loss / max(total_items, 1))

    return (
        total_loss / max(total_items, 1),
        np.asarray(all_true),
        np.asarray(all_pred),
        all_conf,
    )


def safe_qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return 0.0
    value = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return 0.0 if np.isnan(value) else float(value)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0:
        return {"mae": 0.0, "qwk": 0.0, "acc": 0.0, "f1_macro": 0.0}
    return {
        "mae": float(np.abs(y_true - y_pred).mean()),
        "qwk": safe_qwk(y_true, y_pred),
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def format_metrics(prefix: str, loss: float, metrics: dict[str, float]) -> str:
    return (
        f"{prefix} loss={loss:.4f} "
        f"MAE={metrics['mae']:.3f} QWK={metrics['qwk']:.3f} "
        f"Acc={metrics['acc']:.3f} F1={metrics['f1_macro']:.3f}"
    )


def certain_subset_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, confidence: list[str]
) -> dict[str, float]:
    mask = np.array([c == "certo" for c in confidence], dtype=bool)
    return compute_metrics(y_true[mask], y_pred[mask])


def serializable_args(args: argparse.Namespace) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def write_history(run_dir: Path, history: list[dict]) -> None:
    if not history:
        return
    write_json(run_dir / "metrics_history.json", history)
    fieldnames = sorted({key for row in history for key in row})
    with open(run_dir / "metrics_history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    phase: str,
    metrics: dict[str, float],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "phase": phase,
            "metrics": metrics,
            "args": serializable_args(args),
            "class_to_id": CLASS_TO_ID,
        },
        path,
    )


def run_phase(
    phase: str,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: CoralLoss,
    device: torch.device,
    args: argparse.Namespace,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    best_score: tuple[float, float],
    best_path: Path,
    run_dir: Path,
    history: list[dict],
) -> tuple[tuple[float, float], int]:
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        print(f"\n[{phase}] epoch {epoch}/{epochs}")
        train_loss, y_true, y_pred, conf = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler,
            args.grad_accum_steps,
            use_amp,
            desc="train",
        )
        train_metrics = compute_metrics(y_true, y_pred)
        print(format_metrics("train", train_loss, train_metrics))

        with torch.no_grad():
            val_loss, val_true, val_pred, val_conf = run_epoch(
                model,
                val_loader,
                criterion,
                device,
                optimizer=None,
                scaler=None,
                grad_accum_steps=1,
                amp=use_amp,
                desc="val",
            )
        val_metrics = compute_metrics(val_true, val_pred)
        val_certain = certain_subset_metrics(val_true, val_pred, val_conf)
        print(format_metrics("val  ", val_loss, val_metrics))
        print(
            "val/certo "
            f"MAE={val_certain['mae']:.3f} QWK={val_certain['qwk']:.3f} "
            f"Acc={val_certain['acc']:.3f} F1={val_certain['f1_macro']:.3f}"
        )

        score = (val_metrics["qwk"], -val_metrics["mae"])
        improved = score > best_score
        history.append(
            {
                "phase": phase,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_mae": train_metrics["mae"],
                "train_qwk": train_metrics["qwk"],
                "train_acc": train_metrics["acc"],
                "train_f1_macro": train_metrics["f1_macro"],
                "val_loss": val_loss,
                "val_mae": val_metrics["mae"],
                "val_qwk": val_metrics["qwk"],
                "val_acc": val_metrics["acc"],
                "val_f1_macro": val_metrics["f1_macro"],
                "val_certo_mae": val_certain["mae"],
                "val_certo_qwk": val_certain["qwk"],
                "val_certo_acc": val_certain["acc"],
                "val_certo_f1_macro": val_certain["f1_macro"],
                "is_best": improved,
            }
        )
        write_history(run_dir, history)

        if improved:
            best_score = score
            stale_epochs = 0
            save_checkpoint(best_path, model, optimizer, epoch, phase, val_metrics, args)
            write_json(
                run_dir / "best_metrics.json",
                {
                    "phase": phase,
                    "epoch": epoch,
                    "selection": "max val_qwk, tie-break min val_mae",
                    "val_loss": val_loss,
                    "val": val_metrics,
                    "val_certo": val_certain,
                    "checkpoint": str(best_path),
                },
            )
            print(f"[checkpoint] saved {best_path}")
        else:
            stale_epochs += 1

        if args.dry_run:
            print("[dry-run] stopping after one epoch.")
            break
        if stale_epochs >= args.early_stopping:
            print(f"[early-stop] no validation improvement for {stale_epochs} epochs.")
            break

    return best_score, stale_epochs


def evaluate_test(
    model: torch.nn.Module,
    checkpoint_path: Path,
    test_loader: DataLoader,
    criterion: CoralLoss,
    device: torch.device,
    args: argparse.Namespace,
    run_dir: Path,
) -> None:
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])

    use_amp = args.amp and device.type == "cuda"
    with torch.no_grad():
        test_loss, y_true, y_pred, conf = run_epoch(
            model,
            test_loader,
            criterion,
            device,
            optimizer=None,
            scaler=None,
            grad_accum_steps=1,
            amp=use_amp,
            desc="test",
        )
    metrics = compute_metrics(y_true, y_pred)
    certain = certain_subset_metrics(y_true, y_pred, conf)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    print("\n=== Test ===")
    print(format_metrics("test", test_loss, metrics))
    print(
        "test/certo "
        f"MAE={certain['mae']:.3f} QWK={certain['qwk']:.3f} "
        f"Acc={certain['acc']:.3f} F1={certain['f1_macro']:.3f}"
    )
    print("confusion matrix rows=true cols=pred labels=[Baixo, Médio, Alto]")
    print(matrix)

    payload = {
        "test_loss": test_loss,
        "test": metrics,
        "test_certo": certain,
        "labels": [ID_TO_CLASS[i] for i in range(3)],
        "confusion_matrix_rows_true_cols_pred": matrix.tolist(),
        "checkpoint": str(checkpoint_path),
    }
    write_json(run_dir / "test_metrics.json", payload)
    with open(run_dir / "confusion_matrix.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *[ID_TO_CLASS[i] for i in range(3)]])
        for idx, row in enumerate(matrix.tolist()):
            writer.writerow([ID_TO_CLASS[idx], *row])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--annotations", type=Path, default=ROOT / "MapillaryDatasetL+R-Annoted.json"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "training" / "runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--model", default="efficientnet_b2")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=25)
    parser.add_argument("--unfreeze-blocks", type=int, default=2)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-head-lr", type=float, default=3e-4)
    parser.add_argument("--finetune-backbone-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--drop-rate", type=float, default=0.3)
    parser.add_argument(
        "--hflip-p",
        type=float,
        default=0.0,
        help="Horizontal flip probability for training augmentation. Use 0.5 for the flip ablation.",
    )
    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--early-stopping", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--pretrained", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use a tiny subset and stop after one epoch per phase.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    samples = parse_annotations(args.root, args.annotations)
    if args.dry_run:
        samples = samples[:96]
        args.head_epochs = min(args.head_epochs, 1)
        args.finetune_epochs = min(args.finetune_epochs, 1)
        args.early_stopping = 99

    train_samples, val_samples, test_samples = split_samples(
        samples,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    print("\n=== Dataset ===")
    print_split_summary("train", train_samples)
    print_split_summary("val", val_samples)
    print_split_summary("test", test_samples)

    train_tfms = build_transforms(args.image_size, train=True, hflip_p=args.hflip_p)
    eval_tfms = build_transforms(args.image_size, train=False)
    train_loader = make_loader(
        train_samples, train_tfms, args.batch_size, args.num_workers, True, device
    )
    val_loader = make_loader(
        val_samples, eval_tfms, args.batch_size, args.num_workers, False, device
    )
    test_loader = make_loader(
        test_samples, eval_tfms, args.batch_size, args.num_workers, False, device
    )

    print("\n=== Model ===")
    print(
        f"{args.model} | pretrained={args.pretrained} | image_size={args.image_size} | "
        f"batch={args.batch_size} | accum={args.grad_accum_steps} | "
        f"effective_batch={args.batch_size * args.grad_accum_steps} | "
        f"device={device} | amp={args.amp and device.type == 'cuda'} | "
        f"hflip_p={args.hflip_p}"
    )
    model = timm.create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=2,
        drop_rate=args.drop_rate,
    ).to(device)
    criterion = CoralLoss(reduction=None)

    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", serializable_args(args))
    best_path = run_dir / "best.pth"
    best_score = (-float("inf"), -float("inf"))
    history: list[dict] = []

    freeze_for_head_training(model)
    trainable, total = trainable_parameter_count(model)
    print(f"head phase trainable params: {trainable:,}/{total:,}")
    head_optimizer = make_optimizer(
        model,
        backbone_lr=args.head_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
    )
    best_score, _ = run_phase(
        "head",
        model,
        train_loader,
        val_loader,
        criterion,
        device,
        args,
        args.head_epochs,
        head_optimizer,
        best_score,
        best_path,
        run_dir,
        history,
    )

    if args.finetune_epochs > 0:
        unfreeze_last_blocks(model, args.unfreeze_blocks)
        trainable, total = trainable_parameter_count(model)
        print(f"\nfinetune phase trainable params: {trainable:,}/{total:,}")
        finetune_optimizer = make_optimizer(
            model,
            backbone_lr=args.finetune_backbone_lr,
            head_lr=args.finetune_head_lr,
            weight_decay=args.weight_decay,
        )
        best_score, _ = run_phase(
            "finetune",
            model,
            train_loader,
            val_loader,
            criterion,
            device,
            args,
            args.finetune_epochs,
            finetune_optimizer,
            best_score,
            best_path,
            run_dir,
            history,
        )

    evaluate_test(model, best_path, test_loader, criterion, device, args, run_dir)
    print(f"\nRun dir: {run_dir}")


if __name__ == "__main__":
    main()
