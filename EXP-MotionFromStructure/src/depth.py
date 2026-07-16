"""
Depth-Anything-V2 metric outdoor depth wrapper.

Honest caveat: the model was trained on perspective images. Our images are
fisheye (~195° FOV). The model still produces plausible-looking depth on the
central region but its metric scale at the periphery is not guaranteed.
Stage 2 mitigates by fitting the ground plane on `road` pixels (which sit in
the lower-central region of the image, where the perspective approximation is
strongest), then applying that plane to the rest of the frame.

The model is loaded lazily on first use.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np


DEFAULT_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"


@lru_cache(maxsize=2)
def _load_model(model_id: str, device: str):
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    proc = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()
    return proc, model, torch


def predict_depth(image: np.ndarray,
                  model_id: str = DEFAULT_MODEL_ID,
                  device: str = "cuda") -> np.ndarray:
    """
    Run Depth Anything V2 on an HxWx3 uint8 RGB image and return an HxW float32
    metric depth map (meters along Z). Output is upsampled to image size.
    """
    from PIL import Image
    import torch
    proc, model, _torch = _load_model(model_id, device)
    pil = Image.fromarray(image)
    inputs = proc(images=pil, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
        pred = out.predicted_depth  # (1, h, w) at the model's internal scale
        if pred.ndim == 3:
            pred = pred[:, None, :, :]
        upsampled = torch.nn.functional.interpolate(
            pred, size=image.shape[:2], mode="bicubic", align_corners=False
        )
        depth = upsampled[0, 0].cpu().numpy()
    return depth.astype(np.float32)
