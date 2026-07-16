"""
Generate a focused Label Studio review project from model disagreement CSVs.

The output is meant to be imported into a new Label Studio project. Each task
contains the crop image, the original human label, the model prediction, scores,
and the previous task_id so reviewed labels can be merged back later.

Example:
    python label_studio/generate_review_project.py \
      --review-csv training/runs/20260423-190255/review_queue.csv \
      --max-priority 3
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE_SERVER = "http://localhost:8081"


def choice_prediction(label_name: str, score: float, model_version: str) -> list[dict]:
    return [
        {
            "model_version": model_version,
            "score": score,
            "result": [
                {
                    "from_name": "height_class",
                    "to_name": "original",
                    "type": "choices",
                    "value": {"choices": [label_name]},
                }
            ],
        }
    ]


def build_review_text(row: dict) -> str:
    return "\n".join(
        [
            f"Prioridade: {row['review_priority']} - {row['review_reason']}",
            f"Task original: {row['task_id']} | image_id: {row['image_id']} | lado: {row['side']}",
            f"Label humano anterior: {row['human_label']} ({row['human_confidence']})",
            f"Predicao modelo: {row['pred_label']} | score: {row['score']} | margem: {row['margin']}",
            (
                "Probabilidades: "
                f"Baixo={row['prob_baixo']} | Medio={row['prob_medio']} | Alto={row['prob_alto']}"
            ),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=ROOT / "training" / "runs" / "20260423-190255" / "review_queue.csv",
    )
    parser.add_argument("--out-tasks", type=Path, default=ROOT / "label_studio" / "review_tasks.json")
    parser.add_argument(
        "--out-template",
        type=Path,
        default=ROOT / "label_studio" / "review_template.xml",
    )
    parser.add_argument("--image-server", default=DEFAULT_IMAGE_SERVER)
    parser.add_argument("--model-version", default=None)
    parser.add_argument("--max-priority", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_version = args.model_version or args.review_csv.parent.name
    rows = []
    with args.review_csv.open() as f:
        for row in csv.DictReader(f):
            if int(row["review_priority"]) > args.max_priority:
                continue
            rows.append(row)
            if args.limit and len(rows) >= args.limit:
                break

    tasks = []
    for row in rows:
        score = float(row["score"])
        path = row["path"]
        tasks.append(
            {
                "data": {
                    "original": f"{args.image_server}/{path}",
                    "source_task_id": int(row["task_id"]),
                    "image_id": row["image_id"],
                    "side": row["side"],
                    "source_path": path,
                    "review_priority": int(row["review_priority"]),
                    "review_reason": row["review_reason"],
                    "human_label": row["human_label"],
                    "human_confidence": row["human_confidence"],
                    "pred_label": row["pred_label"],
                    "pred_score": score,
                    "pred_margin": float(row["margin"]),
                    "prob_baixo": float(row["prob_baixo"]),
                    "prob_medio": float(row["prob_medio"]),
                    "prob_alto": float(row["prob_alto"]),
                    "review_text": build_review_text(row),
                },
                "predictions": choice_prediction(row["pred_label"], score, model_version),
            }
        )

    args.out_tasks.parent.mkdir(parents=True, exist_ok=True)
    args.out_tasks.write_text(json.dumps(tasks, indent=2, ensure_ascii=False))
    args.out_template.write_text(REVIEW_TEMPLATE)

    print(f"Tasks: {len(tasks)}")
    print(f"Saved: {args.out_tasks}")
    print(f"Saved: {args.out_template}")


REVIEW_TEMPLATE = """<View>
  <View style="position:sticky; top:0; z-index:100; background:#111; color:#fff; padding:12px; border-radius:8px; margin-bottom:10px;">
    <Header value="Revisao assistida por modelo"/>
    <Text name="review_meta" value="$review_text"/>

    <Header value="Altura revisada:"/>
    <Choices name="height_class" toName="original" choice="single" required="true" showInLine="true">
      <Choice value="Baixo" hotkey="1" hint="Rasteira / acamada - quase no nivel do solo"/>
      <Choice value="Médio" hotkey="2" hint="Ereta mas contida - visivel na faixa lateral"/>
      <Choice value="Alto"  hotkey="3" hint="Volumosa e ereta - domina a faixa de dominio"/>
    </Choices>

    <Header value="Confianca revisada:"/>
    <Choices name="confidence" toName="original" choice="single" required="true" showInLine="true">
      <Choice value="certo"   hotkey="q" hint="Tenho certeza - nao e caso limite"/>
      <Choice value="incerto" hotkey="w" hint="Caso limite - poderia ser a classe adjacente"/>
    </Choices>
  </View>

  <Image name="original" value="$original" zoom="true" zoomControl="true"/>

  <View style="background:#1c1c1c; color:#fff; border-radius:8px; padding:12px; margin-top:12px;">
    <Header value="REGRA DE REVISAO"/>
    <Header value="Rotule pela condicao predominante da faixa lateral util, nao pelo pior ponto isolado."/>
    <Header value="BAIXO: maioria rasteira/aparada; tufos altos pontuais nao dominam."/>
    <Header value="MEDIO: acima do rasteiro, mas sem massa alta continua dominante."/>
    <Header value="ALTO: vegetacao alta, volumosa e continua domina a lateral."/>
    <Header value="Se a decisao depender da fronteira entre classes adjacentes, use incerto."/>
  </View>
</View>
"""


if __name__ == "__main__":
    main()
