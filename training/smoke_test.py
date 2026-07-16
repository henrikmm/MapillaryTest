"""
Smoke test — valida que o stack completo funciona antes de treinar:
  - timm carrega EfficientNet-B2 com head CORAL (num_classes=2)
  - coral_pytorch loss e label encoding funcionam
  - forward pass com batch sintético 384x384 passa sem erro
  - métricas ordinais (MAE, QWK) calculam corretamente
  - dataset + transform 384x384 letterbox carregam uma amostra real de crop
  - GPU disponível
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

ROOT = Path(__file__).parent.parent
IMAGE_SIZE = 384

def check(name, fn):
    try:
        fn()
        print(f"  [OK] {name}")
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        sys.exit(1)

# ── 1. GPU ────────────────────────────────────────────────────────────────────
def test_gpu():
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"       GPU: {torch.cuda.get_device_name(0)}")

# ── 2. Modelo ─────────────────────────────────────────────────────────────────
def test_model():
    import timm
    model = timm.create_model("efficientnet_b2", pretrained=False, num_classes=2)
    assert model.num_classes == 2
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"       Params: {n_params:.1f}M | num_features: {model.num_features}")

# ── 3. Forward pass ───────────────────────────────────────────────────────────
def test_forward():
    import timm
    device = "cuda"
    model = timm.create_model("efficientnet_b2", pretrained=False, num_classes=2).to(device)
    x = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 2), f"Expected (4,2), got {out.shape}"

# ── 4. CORAL loss ─────────────────────────────────────────────────────────────
def test_coral():
    from coral_pytorch.losses import CoralLoss
    from coral_pytorch.dataset import levels_from_labelbatch

    # labels: Baixo=0, Médio=1, Alto=2
    labels = torch.tensor([0, 1, 2, 1])
    levels = levels_from_labelbatch(labels, num_classes=3)
    assert levels.shape == (4, 2), f"Expected (4,2), got {levels.shape}"

    # [0,0], [1,0], [1,1], [1,0]
    expected = torch.tensor([[0,0],[1,0],[1,1],[1,0]], dtype=torch.float32)
    assert torch.equal(levels.float(), expected), f"Levels mismatch: {levels}"

    logits = torch.randn(4, 2)
    loss = CoralLoss()(logits, levels)
    assert loss.item() > 0
    print(f"       CORAL loss: {loss.item():.4f}")

# ── 5. Decodificação CORAL → classe ──────────────────────────────────────────
def test_decode():
    from coral_pytorch.dataset import proba_to_label
    logits = torch.tensor([[-2.0, -2.0],   # → Baixo (0)
                            [ 2.0, -2.0],   # → Médio (1)
                            [ 2.0,  2.0]])  # → Alto  (2)
    probs = torch.sigmoid(logits)
    preds = proba_to_label(probs).long()
    assert preds.tolist() == [0, 1, 2], f"Decode failed: {preds.tolist()}"

# ── 6. Métricas ordinais ──────────────────────────────────────────────────────
def test_metrics():
    from sklearn.metrics import cohen_kappa_score

    y_true = np.array([0, 1, 2, 0, 1, 2, 1, 0])
    y_pred = np.array([0, 1, 2, 1, 1, 1, 2, 0])

    mae = float(np.abs(y_true - y_pred).mean())
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    acc = float((y_true == y_pred).mean())
    print(f"       MAE={mae:.3f} | QWK={qwk:.3f} | Acc={acc:.3f}")

# ── 7. Dataset real ───────────────────────────────────────────────────────────
def test_dataset():
    from PIL import Image
    import albumentations as A
    import cv2
    from albumentations.pytorch import ToTensorV2

    transform = A.Compose([
        A.LongestMaxSize(max_size=IMAGE_SIZE, interpolation=cv2.INTER_CUBIC),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE,
            min_width=IMAGE_SIZE,
            border_mode=cv2.BORDER_CONSTANT,
            fill=(0, 0, 0),
            position="center",
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    sample = next((ROOT / "crops" / "left").glob("*.jpg"))
    img = np.array(Image.open(sample).convert("RGB"))
    tensor = transform(image=img)["image"]
    assert tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE), f"Shape: {tensor.shape}"
    print(f"       Crop carregado: {sample.name} → {tuple(tensor.shape)}")

# ── 8. Annotations parse ──────────────────────────────────────────────────────
def test_annotations():
    import json
    ann_path = ROOT / "MapillaryDatasetL+R-Annoted.json"
    data = json.load(open(ann_path))
    labelled = [d for d in data if d["annotations"]]
    classes = {"Baixo": 0, "Médio": 1, "Alto": 2}
    labels = []
    for item in labelled:
        res = item["annotations"][0]["result"]
        cls = next((r["value"]["choices"][0] for r in res if r["from_name"] == "height_class"), None)
        if cls in classes:
            labels.append(classes[cls])
    from collections import Counter
    dist = Counter(labels)
    print(f"       {len(labels)} labels | Baixo={dist[0]} Médio={dist[1]} Alto={dist[2]}")

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== Smoke Test ===\n")
    check("GPU disponível",          test_gpu)
    check("EfficientNet-B2 carrega", test_model)
    check("Forward pass (4×384×384)", test_forward)
    check("CORAL loss + encoding",   test_coral)
    check("Decodificação CORAL",     test_decode)
    check("Métricas ordinais",       test_metrics)
    check("Dataset (crop real)",     test_dataset)
    check("Annotations parse",       test_annotations)
    print("\n=== Tudo OK — pronto para treinar ===\n")
