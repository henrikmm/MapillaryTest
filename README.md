# Mapillary Grass Height Dataset

Projeto experimental para construir um dataset e treinar um modelo de visão
computacional capaz de classificar a altura visual da grama em faixas laterais
de rodovias.

A unidade principal do dataset é um crop lateral da imagem, separado por lado
esquerdo e direito da via, com rótulo ordinal de altura:

| Classe | Rótulo | Interpretação visual |
| --- | --- | --- |
| 1 | Baixo | Grama rasteira, acamada ou quase no nível do solo |
| 2 | Médio | Grama ereta e visível, mas ainda contida |
| 3 | Alto | Grama volumosa, alta e dominante na lateral |

O objetivo não é medir centímetros diretamente a partir de uma imagem monocular.
O que se modela aqui é uma percepção visual ordinal calibrada por exemplos
anotados.

## Estado Atual do Dataset

- 999 metadados de imagens em `metadata.jsonl`
- 999 máscaras binárias de terreno em `helpers/masks/`
- 995 crops do lado esquerdo em `crops/left/`
- 992 crops do lado direito em `crops/right/`
- 1987 crops anotados em `MapillaryDatasetL+R-Annoted.json`
- 997 `image_id`s únicos com pelo menos um lado anotado

Distribuição dos rótulos anotados:

| Classe | Quantidade |
| --- | ---: |
| Baixo | 705 |
| Médio | 896 |
| Alto | 386 |

Distribuição de confiança:

| Confiança | Quantidade |
| --- | ---: |
| certo | 1122 |
| incerto | 865 |

## Estrutura do Repositório

```text
MapillaryTest/
├── images/                         # Frames originais baixados do Mapillary
├── crops/
│   ├── left/                       # Crops anotados do lado esquerdo
│   └── right/                      # Crops anotados do lado direito
├── helpers/
│   ├── masks/                      # Máscaras binárias terrain-only
│   ├── segmented/                  # Visualizações auxiliares de segmentação
│   ├── explore_crop_features.py    # Extração exploratória de features
│   └── exploratory_crop_features.csv
├── label_studio/
│   ├── generate_crops.py           # Gera crops e tasks para anotação
│   ├── generate_tasks.py           # Fluxo legado para imagens inteiras
│   ├── image_server.py             # Servidor local de imagens
│   ├── start.sh                    # Inicialização auxiliar do Label Studio
│   └── template.xml                # Interface de anotação
├── training/
│   ├── TRAINING_PLAN.md            # Plano detalhado de treinamento
│   └── smoke_test.py               # Verificação do stack de treino
├── download.py                     # Download da sequência Mapillary
├── mask_batch.py                   # Geração das máscaras terrain-only
├── segment.py                      # Segmentação/visualização de uma imagem
├── segment_batch.py                # Visualizações segmentadas em lote
├── MapillaryDatasetL+R-Annoted.json
└── metadata.jsonl
```

## Pipeline

O fluxo atual é crop-based:

1. Baixar imagens e metadados de uma sequência Mapillary.
2. Rodar segmentação semântica para identificar regiões de terreno/grama.
3. Gerar máscaras binárias `terrain-only`.
4. Separar cada imagem em lado esquerdo e direito.
5. Apagar pixels que não pertencem à máscara de terreno.
6. Recortar o bounding box da grama de cada lado, com padding.
7. Anotar os crops no Label Studio.
8. Treinar um classificador ordinal com os crops e o JSON anotado.

## Extração do Dataset via Mapillary

O script `download.py` usa a Graph API do Mapillary para baixar uma sequência de
imagens e seus metadados.

### 1. Configurar credenciais

Crie um `.env` com:

```env
MAPILLARY_ACCESS_TOKEN=seu_token
MAPILLARY_SEQUENCE_KEY=id_da_sequencia
MAPILLARY_API_BASE=https://graph.mapillary.com
```

### 2. Buscar IDs da sequência

O script consulta:

```text
GET /image_ids?sequence_id={MAPILLARY_SEQUENCE_KEY}
```

Isso retorna os `image_id`s pertencentes à sequência. O script percorre a
paginação da API até coletar todos os IDs.

### 3. Buscar metadados por imagem

Para cada `image_id`, o script consulta campos como:

- `id`
- `sequence`
- `captured_at`
- `geometry`
- `compass_angle`
- `is_pano`
- `width`
- `height`
- `altitude`
- `camera_type`
- URLs de thumbnails em diferentes resoluções

Esses metadados são gravados incrementalmente em `metadata.jsonl`.

### 4. Baixar os arquivos de imagem

O script escolhe uma URL de imagem a partir do argumento `--resolution`:

| Argumento | Campo Mapillary usado |
| --- | --- |
| `256` | `thumb_256_url` |
| `1024` | `thumb_1024_url` |
| `2048` | `thumb_2048_url` |
| `original` | `thumb_original_url` |

Uso típico:

```bash
python download.py --resolution original --workers 6
```

As imagens são salvas em `images/{image_id}.jpg`.

### 5. Retomada de download

Se `metadata.jsonl` já existe, o script lê os IDs já processados e baixa apenas
o que falta. Isso permite retomar a extração sem recomeçar do zero.

## Segmentação com NVIDIA SegFormer

A segmentação usa o modelo:

```text
nvidia/segformer-b1-finetuned-cityscapes-1024-1024
```

Esse é um SegFormer-B1 ajustado no dataset Cityscapes para segmentação semântica
urbana. A escolha faz sentido como etapa auxiliar porque Cityscapes possui
classes compatíveis com cenas rodoviárias, incluindo `road`, `terrain`,
`vegetation`, `car`, `bus`, `truck`, `traffic sign` e outras.

Neste projeto, a segmentação não é o modelo final. Ela é usada para preparar o
dataset de classificação de altura.

### Visualização de uma imagem

`segment.py` segmenta uma imagem e gera uma visualização colorida das classes
selecionadas:

```bash
python segment.py images/927311799628456.jpg
```

Por padrão, o resultado é salvo em `helpers/segmented/`.

### Visualização em lote

`segment_batch.py` processa todas as imagens em `images/` e salva imagens
auxiliares em `helpers/segmented/`, destacando principalmente regiões de
`vegetation` e `terrain`:

```bash
python segment_batch.py --batch-size 4 --workers 4
```

Essas visualizações ajudam na inspeção humana, mas não são o ground truth do
treinamento.

### Máscaras binárias para geração de crops

O fluxo efetivamente usado para os crops finais está em `mask_batch.py`.

Ele gera máscaras binárias em:

```text
helpers/masks/{image_id}_mask.png
```

Nessas máscaras:

- `255` significa terreno/grama
- `0` significa qualquer outra coisa

Uso:

```bash
python mask_batch.py --batch-size 4 --workers 4
```

O script usa apenas a classe `terrain` do Cityscapes. A classe `vegetation` foi
deixada fora desse passo porque tende a incluir árvores e arbustos, o que pode
contaminar a região que será anotada como grama lateral.

## Geração dos Crops

Os crops são gerados por `label_studio/generate_crops.py`.

Para cada imagem:

1. Carrega o frame original em `images/{image_id}.jpg`.
2. Carrega a máscara em `helpers/masks/{image_id}_mask.png`.
3. Divide a imagem verticalmente em lado esquerdo e direito.
4. Verifica se há pixels de grama suficientes no lado analisado.
5. Apaga todos os pixels fora da máscara, deixando fundo preto.
6. Calcula o bounding box da grama daquele lado.
7. Adiciona padding ao redor da região.
8. Salva o crop em `crops/left/` ou `crops/right/`.
9. Gera `label_studio/tasks.json` apontando para cada crop.

Uso:

```bash
python label_studio/generate_crops.py
```

Cada amostra anotada representa o par:

```text
(image_id, side)
```

Exemplo:

```text
crops/left/927311799628456.jpg
crops/right/927311799628456.jpg
```

## Anotação no Label Studio

O arquivo `label_studio/template.xml` define dois grupos de escolha:

- Altura da grama:
  - `Baixo`
  - `Médio`
  - `Alto`
- Confiança:
  - `certo`
  - `incerto`

O resultado exportado do Label Studio está em:

```text
MapillaryDatasetL+R-Annoted.json
```

Esse JSON é o ground truth atual do projeto.

## Treinamento Pretendido

O plano de treinamento está detalhado em `training/TRAINING_PLAN.md`.

A recomendação atual é treinar um classificador ordinal crop-only usando:

- PyTorch
- `timm`
- EfficientNet-B2 como backbone
- `coral_pytorch` para regressão ordinal
- `albumentations` para preprocessing e augmentation
- `scikit-learn` para splits e métricas

### Por que tratar como problema ordinal

As classes têm ordem natural:

```text
Baixo < Médio < Alto
```

Errar `Baixo` como `Médio` é menos grave do que errar `Baixo` como `Alto`.
Por isso, o plano propõe CORAL em vez de uma classificação softmax comum.

Com 3 classes, CORAL usa `K - 1` saídas:

```text
Baixo -> [0, 0]
Médio -> [1, 0]
Alto  -> [1, 1]
```

### Métricas recomendadas

- MAE ordinal
- Quadratic Weighted Kappa, ou QWK
- Accuracy
- F1 macro
- Matriz de confusão

MAE e QWK são as métricas mais importantes porque respeitam a distância entre
classes ordinais.

### Split correto

O split deve ser feito por `image_id`, não por arquivo de crop.

Isso evita que o lado esquerdo de uma imagem caia no treino e o lado direito da
mesma imagem caia na validação ou teste.

### Preprocessing recomendado

Os crops têm dimensões variáveis. O plano recomenda:

1. Redimensionar preservando aspect ratio.
2. Aplicar padding preto até um tamanho fixo.
3. Normalizar.

Tamanhos sugeridos para comparação:

- `320x320`
- `384x384`

Augmentations geométricas fortes devem ser evitadas, porque podem alterar pistas
visuais importantes de altura e volume da vegetação.

### Treino do baseline

O script de treino está em:

```text
training/train.py
```

Configuração padrão:

- EfficientNet-B2
- CORAL com 2 saídas ordinais
- `384x384` com padding
- split 70/15/15 por `image_id`
- `certo = 1.0` e `incerto = 0.6`
- `batch_size = 8`
- `grad_accum_steps = 2`
- AMP ligado em CUDA

Para treinar na RTX 4060 Ti 8 GB:

```bash
python training/train.py
```

Se couber com folga:

```bash
python training/train.py --batch-size 16 --grad-accum-steps 1
```

Se der falta de memória:

```bash
python training/train.py --batch-size 4 --grad-accum-steps 4
```

Para testar horizontal flip como augmentation:

```bash
python training/train.py \
  --annotations MapillaryDatasetL+R-Annoted-v2.json \
  --batch-size 16 \
  --grad-accum-steps 1 \
  --hflip-p 0.5 \
  --run-name v2-hflip-p05-seed42
```

O histórico dos experimentos está em:

```text
training/EXPERIMENTS.md
```

Para validar o pipeline sem baixar pesos pretrained nem usar GPU:

```bash
python training/train.py --dry-run --no-pretrained --device cpu --num-workers 0
```

### Exportar previsões para revisão no Label Studio

Depois de treinar, gere predictions para as tasks existentes do Label Studio:

```bash
python training/export_label_studio_predictions.py \
  --checkpoint training/runs/20260423-190255/best.pth
```

O script cria, dentro da pasta do run:

```text
label_studio_predictions.json  # payload para import/predictions
predictions.csv                # todas as previsões
review_queue.csv               # fila priorizada de revisão
```

Para importar direto via API:

```bash
export LABEL_STUDIO_URL=http://localhost:8080
export LABEL_STUDIO_PROJECT_ID=13
export LABEL_STUDIO_API_TOKEN=seu_token

python training/export_label_studio_predictions.py \
  --checkpoint training/runs/20260423-190255/best.pth \
  --upload
```

O script usa `task_id`s do arquivo exportado pelo Label Studio. Portanto, as
previsões devem ser importadas no mesmo projeto de onde saiu
`MapillaryDatasetL+R-Annoted.json`.

### Projeto de revisão rápida

Para evitar alternar entre CSV e Label Studio, gere uma fila de revisão como um
novo projeto:

```bash
python label_studio/generate_review_project.py \
  --review-csv training/runs/20260423-190255/review_queue.csv \
  --max-priority 3
```

Isso gera:

```text
label_studio/review_tasks.json
label_studio/review_template.xml
```

Crie um novo projeto no Label Studio, cole o conteúdo de
`label_studio/review_template.xml` como labeling config e importe
`label_studio/review_tasks.json`. Cada task mostra o label anterior, a predição,
o score, o motivo da revisão e o crop.

Depois de revisar e exportar esse projeto, gere uma nova versão do dataset:

```bash
python label_studio/merge_review_annotations.py \
  --review-export ReviewProjectExport.json \
  --out MapillaryDatasetL+R-Annoted-v2.json
```

## Arquivos Core para Treino

Para treinar o modelo principal, os arquivos essenciais são:

```text
crops/left/*.jpg
crops/right/*.jpg
MapillaryDatasetL+R-Annoted.json
```

Os demais artefatos são auxiliares para download, segmentação, geração de crops,
inspeção ou experimentação.

## Smoke Test

`training/smoke_test.py` valida o stack esperado de treinamento:

- GPU/CUDA disponível
- EfficientNet-B2 via `timm`
- forward pass
- `CoralLoss`
- encoding/decoding CORAL
- métricas ordinais
- leitura de crop real
- parse do JSON anotado

Uso no ambiente de treino:

```bash
python training/smoke_test.py
```

Esse teste deve ser lido como verificação do ambiente onde o treinamento será
executado.
