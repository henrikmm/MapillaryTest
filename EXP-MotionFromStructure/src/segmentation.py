"""
Cityscapes-pretrained SegFormer wrapper.

Class mapping notes (confirmed with user, ASSUMPTIONS.md A6):
  * `terrain` (id 9) is what captures GRASS in this dataset — the actual
    measurement target.
  * `vegetation` (id 8) tends to capture trees / large bushes. Useful but
    secondary.
  * `road` (id 0) is the surface used to fit the ground plane.

The model is loaded lazily on first use.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np


CITYSCAPES_CLASSES = {
    "road": 0, "sidewalk": 1, "building": 2, "wall": 3, "fence": 4,
    "pole": 5, "traffic_light": 6, "traffic_sign": 7,
    "vegetation": 8, "terrain": 9, "sky": 10,
    "person": 11, "rider": 12, "car": 13, "truck": 14, "bus": 15,
    "train": 16, "motorcycle": 17, "bicycle": 18,
}

DEFAULT_MODEL_ID = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"


@lru_cache(maxsize=2)
def _load_model(model_id: str, device: str):
    import torch
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
    proc = AutoImageProcessor.from_pretrained(model_id)
    model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(device).eval()
    return proc, model, torch


def segment_classmap(image: np.ndarray, classes: Iterable[str],
                     model_id: str = DEFAULT_MODEL_ID,
                     device: str = "cuda") -> dict[str, np.ndarray]:
    """
    Return a dict {class_name: HxW bool mask} for the requested Cityscapes
    classes, evaluated on `image` (H, W, 3 uint8 RGB).
    """
    from PIL import Image
    proc, model, torch = _load_model(model_id, device)
    pil = Image.fromarray(image)
    inputs = proc(images=pil, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits  # (1, C, h, w) at the model's internal scale
        # Upsample to original image resolution.
        upsampled = torch.nn.functional.interpolate(
            logits, size=image.shape[:2], mode="bilinear", align_corners=False
        )
        labels = upsampled.argmax(dim=1)[0].cpu().numpy()  # (H, W) int64
    return {c: (labels == CITYSCAPES_CLASSES[c]) for c in classes}
