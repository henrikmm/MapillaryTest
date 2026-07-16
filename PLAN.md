# Projeto: Classificação de Altura de Grama em Faixa de Domínio Rodoviário

## Objetivo
Treinar um modelo de visão computacional para classificar a altura visual da
grama lateral de rodovias em 3 classes ordinais:

| Classe | Label | Interpretação visual |
|--------|-------|----------------------|
| 1 | Baixo | grama rasteira, acamada, próxima do solo |
| 2 | Médio | grama ereta, visível, mas ainda contida |
| 3 | Alto | grama volumosa, ereta, dominante na lateral |

> Nota de honestidade: os valores em cm são uma referência operacional da
> concessionária. A partir de imagem monocular, o que estamos modelando é uma
> percepção visual ordinal calibrada por exemplos âncora.

---

## O Que É Core no Projeto

### Unidade real de anotação
- A anotação foi feita **somente nos crops** em `crops/left/*.jpg` e
  `crops/right/*.jpg`.
- O ground truth do treino é o arquivo `MapillaryDatasetL+R-Annoted.json`
  associado a esses crops.
- Cada item anotado corresponde a um par `(image_id, side)`.

### O que entra no treino
- `crops/left/*.jpg`
- `crops/right/*.jpg`
- `MapillaryDatasetL+R-Annoted.json`

### O que é auxiliar
- `helpers/masks/`:
  máscaras binárias `terrain` usadas para gerar os crops.
- `helpers/segmented/`:
  visualizações auxiliares de segmentação, úteis para inspeção humana, mas não
  usadas como entrada do modelo final.
- `helpers/explore_crop_features.py` e
  `helpers/exploratory_crop_features.csv`:
  exploração manual; não foram usados para anotar o dataset e não fazem parte
  do pipeline principal de treino.

---

## Estrutura Atual

```text
MapillaryTest/
├── images/                         ← frames originais
├── crops/
│   ├── left/                       ← crops anotados do lado esquerdo
│   └── right/                      ← crops anotados do lado direito
├── helpers/
│   ├── masks/                      ← máscaras terrain-only para gerar crops
│   ├── segmented/                  ← visualizações auxiliares
│   ├── explore_crop_features.py    ← exploração opcional
│   └── exploratory_crop_features.csv
├── label_studio/
│   ├── generate_crops.py           ← gera crops + tasks do fluxo real
│   ├── generate_tasks.py           ← utilitário legado para frames originais
│   └── template.xml
├── training/
│   └── TRAINING_PLAN.md            ← plano detalhado de treino e estudo
├── MapillaryDatasetL+R-Annoted.json
├── download.py
├── mask_batch.py
├── segment.py
└── segment_batch.py
```

---

## Pipeline Resumido
1. `download.py`
   Baixa imagens e metadados da sequência Mapillary.
2. `mask_batch.py`
   Gera máscaras binárias `terrain` only em `helpers/masks/`.
3. `label_studio/generate_crops.py`
   Divide em left/right, zera não-grama, recorta bbox com padding e gera os
   crops usados na anotação.
4. Label Studio
   Anota cada crop com `Baixo`, `Médio` ou `Alto`, além de `certo/incerto`.
5. `training/TRAINING_PLAN.md`
   Define o treino do classificador ordinal.

---

## Estado Atual
- 999 imagens originais.
- 1987 crops anotados.
- 997 `image_id`s com pelo menos um lado anotado.
- O fluxo canônico de anotação e treino agora é crop-based.

---

## Próximo Documento
O plano detalhado de implementação, escolha de framework, fine-tuning,
validação, métricas, resolução e tratamento de `certo/incerto` está em:

`training/TRAINING_PLAN.md`
