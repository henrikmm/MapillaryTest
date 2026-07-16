"""
Gerador legado de tasks para frames originais.
O fluxo atual de anotação do treino usa `generate_crops.py`, que cria tasks por
crop em `crops/left` e `crops/right`.
"""

import json, random
from pathlib import Path

ROOT = Path(__file__).parent.parent
IMAGES_DIR = ROOT / "images"

# Serve images via simple HTTP server running on port 8081
IMAGE_SERVER = "http://localhost:8081"

tasks = []
for img_path in sorted(IMAGES_DIR.glob("*.jpg")):
    tasks.append({
        "data": {
            "image_id": img_path.stem,
            "original": f"{IMAGE_SERVER}/images/{img_path.name}",
        }
    })

random.seed(42)
random.shuffle(tasks)

out = ROOT / "label_studio" / "tasks.json"
out.write_text(json.dumps(tasks, indent=2))
print(f"Generated {len(tasks)} tasks → {out}")
