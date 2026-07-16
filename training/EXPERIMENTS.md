# Experimentos

Registro dos testes de treinamento do classificador ordinal de altura da grama.

## Configuração Base

```text
Backbone: EfficientNet-B2
Pretraining: ImageNet
Formulação: CORAL ordinal regression
Loss: CoralLoss
Imagem: 384x384 com aspect ratio preservado + padding preto
Batch: 16
Gradient accumulation: 1
AMP: ligado em CUDA
Split: 70/15/15 agrupado por image_id
Métricas principais: MAE ordinal + QWK
```

Observação: as comparações entre datasets v1 e v2 são indicativas, mas não
perfeitamente controladas, porque mudanças de rótulo podem alterar a
estratificação dos grupos. Comparações de augmentations no mesmo dataset e seed
são mais diretas.

## Linha do Tempo

### 1. Construção do dataset

- Baixamos uma sequência Mapillary com `download.py`.
- Geramos máscaras semanticamente usando
  `nvidia/segformer-b1-finetuned-cityscapes-1024-1024`.
- A primeira exploração com `terrain + vegetation` mostrou risco de capturar
  árvores e arbustos.
- O pipeline final de crops passou a usar apenas `terrain`.
- Os crops foram separados em `left` e `right`, com o fundo fora da máscara
  apagado para preto.
- As anotações foram feitas no Label Studio sobre os crops, não sobre a imagem
  original inteira.

### 2. Baseline v1

- Implementamos EfficientNet-B2 + CORAL.
- Consolidamos `384x384` com aspect ratio preservado e padding preto.
- Rodamos o primeiro baseline em `MapillaryDatasetL+R-Annoted.json`.
- Resultado inicial já foi tecnicamente promissor:
  - `test QWK=0.739`
  - `test MAE=0.281`
  - poucos erros de 2 classes.

### 3. Revisão assistida por modelo

- Exportamos previsões do modelo para achar discordâncias.
- As discordâncias severas indicaram inconsistências reais de rótulo.
- Criamos um fluxo de revisão no Label Studio com `review_tasks.json`.
- Os dois casos de discordância de 2 classes foram revisados para `Médio`.
- Geramos `MapillaryDatasetL+R-Annoted-v2.json`.

### 4. Baseline v2

- Treinamos novamente sobre o dataset revisado.
- O v2 melhorou o conjunto completo e, principalmente, o subconjunto `certo`.
- O maior ganho foi na consistência da classe `Médio`.

### 5. Ablation com HorizontalFlip

- Testamos `HorizontalFlip(p=0.5)` no dataset v2.
- A hipótese era reduzir viés entre lado esquerdo e direito.
- O resultado melhorou casos `certo`, mas piorou o conjunto completo.
- Decisão atual: não usar `p=0.5` como default.

## Estado Atual

Melhor configuração até agora:

```text
Dataset: MapillaryDatasetL+R-Annoted-v2.json
Modelo: EfficientNet-B2 + CORAL
Imagem: 384x384 letterbox
Augmentation: fotométrica leve, sem HorizontalFlip
Batch: 16
AMP: ligado
```

Resultado atual:

```text
test MAE=0.247
test QWK=0.751
test Acc=0.757
test F1=0.757

test/certo MAE=0.155
test/certo QWK=0.845
test/certo Acc=0.851
test/certo F1=0.840
```

Leitura de viabilidade:

- O baseline é tecnicamente viável para classificar a altura visual ordinal da
  grama.
- O modelo aprendeu sinal útil, não apenas ruído: QWK acima de `0.75` em teste
  é forte para um dataset pequeno, ordinal e com rótulos ambíguos.
- Os erros são majoritariamente entre classes vizinhas.
- A classe `Médio` continua sendo o eixo crítico da tarefa.
- O subconjunto `certo` indica que, quando o rótulo humano é limpo, o modelo
  performa em nível bem mais alto.

## Hipótese Aberta - Qualidade dos Crops

Há evidência de que parte dos crops tem pouca informação visual útil, causada
por segmentação ruim, máscara pequena ou recorte muito preto.

Medição rápida em `MapillaryDatasetL+R-Annoted-v2.json`, usando pixels não
pretos como proxy de área visível:

```text
n=1987 crops
q01 área visível:  3.87%
q05 área visível:  9.53%
q10 área visível: 12.94%
q25 área visível: 19.28%
q50 área visível: 27.92%

visible <  3%:  15 crops
visible <  5%:  27 crops
visible <  8%:  63 crops
visible < 10%: 110 crops
visible < 15%: 270 crops
visible < 20%: 545 crops
```

Interpretação:

- A cauda de baixa área visível é real.
- Esses casos provavelmente tornam `Baixo` e `Médio` menos estáveis.
- Antes de melhorar arquitetura, vale testar filtros simples de qualidade.

Experimentos sugeridos:

```text
Exp 04a: remover crops com área visível < 5%
Exp 04b: remover crops com área visível < 10%
Exp 04c: manter crops, mas reduzir peso de amostras com área visível < 10%
Exp 04d: criar fila Label Studio só para crops com área visível < 10%
```

Critério de decisão:

- Manter filtro se melhorar QWK/MAE no teste sem reduzir demais cobertura.
- Observar especialmente se `Baixo -> Médio` e `Médio -> Baixo` melhoram.
- Se remover muitos casos `Baixo`, compensar com revisão ou novo dataset.

## Hipótese Aberta - Generalização Em Novo Dataset

Baixar uma sequência Mapillary completamente nova é o próximo teste mais
importante para saber se o modelo aprendeu altura visual ou apenas o domínio
específico da sequência atual.

Workflow recomendado:

```text
1. Baixar nova sequência Mapillary.
2. Rodar o mesmo pipeline de masks e crops.
3. Rodar inferência com o melhor checkpoint v2.
4. Criar review_tasks apenas para uma amostra estratificada.
5. Anotar manualmente uma parte pequena do novo dataset.
6. Medir performance zero-shot do modelo v2.
7. Só depois decidir se faz fine-tuning com dados novos.
```

Amostra mínima sugerida:

```text
100 a 200 crops anotados manualmente
- balancear left/right
- incluir casos claramente baixo, médio e alto
- incluir crops ruins/duvidosos
```

Pergunta que esse teste responde:

```text
O modelo generaliza para outra estrada/iluminação/câmera/vegetação sem novo
treino?
```

## Handoff Para Próxima Sessão

Ordem recomendada para continuar:

1. Congelar os splits em arquivo para que v1, v2, filtros e augmentations sejam
   comparados exatamente nos mesmos `image_id`s.
2. Implementar uma métrica de qualidade do crop:
   - `visible_ratio`
   - talvez altura/largura do bbox
   - talvez área da máscara original.
3. Rodar `Exp 04a` e `Exp 04b`:
   - v2 sem crops `<5%`
   - v2 sem crops `<10%`
4. Gerar uma fila de revisão Label Studio dos piores crops:
   - `visible_ratio < 10%`
   - priorizar os que são `certo`, porque talvez sejam falsamente confiáveis.
5. Baixar uma nova sequência Mapillary e rodar o pipeline de inferência.
6. Fazer um mini conjunto de teste externo com 100-200 crops.
7. Só depois pesquisar mudanças maiores:
   - segmentação melhor;
   - usar crop original com máscara como canal auxiliar;
   - modelo multi-input crop + contexto;
   - CORN vs CORAL;
   - grouped CV.

Coisas a estudar/pesquisar:

- Como calibrar probabilidades em CORAL.
- Como avaliar datasets ordinais com rótulos incertos.
- Estratégias de active learning em Label Studio.
- Filtros automáticos de qualidade para segmentação.
- Comparação de zero-shot em novo domínio antes de fine-tuning.

## Exp 01 - Dataset v1, Baseline Sem Flip

Comando:

```bash
python training/train.py \
  --batch-size 16 \
  --grad-accum-steps 1
```

Dataset:

```text
MapillaryDatasetL+R-Annoted.json
```

Melhor validação:

```text
finetune epoch 17
val MAE=0.255
val QWK=0.748
val Acc=0.748
val F1=0.737
val/certo QWK=0.821
```

Teste:

```text
test MAE=0.281
test QWK=0.739
test Acc=0.722
test F1=0.728

test/certo MAE=0.213
test/certo QWK=0.812
test/certo Acc=0.787
test/certo F1=0.788
```

Matriz de confusão:

```text
             Pred Baixo  Pred Médio  Pred Alto
True Baixo       80          29          0
True Médio       20          88         22
True Alto         1          11         48
```

Leitura:

- Baseline já forte.
- Quase todos os erros são entre classes adjacentes.
- Classe `Médio` era o principal ponto de ruído.

## Exp 02 - Dataset v2 Revisado, Baseline Sem Flip

Comando:

```bash
python training/train.py \
  --annotations MapillaryDatasetL+R-Annoted-v2.json \
  --batch-size 16 \
  --grad-accum-steps 1
```

Dataset:

```text
MapillaryDatasetL+R-Annoted-v2.json
```

Mudança de dataset:

- Revisão assistida por modelo dos casos prioritários.
- Os dois casos de discordância de 2 classes foram movidos para `Médio`.
- A revisão parece ter tornado a classe `Médio` mais consistente.

Melhor validação:

```text
finetune epoch 14
val MAE=0.215
val QWK=0.783
val Acc=0.789
val F1=0.781

val/certo MAE=0.149
val/certo QWK=0.857
val/certo Acc=0.851
val/certo F1=0.845
```

Teste:

```text
test MAE=0.247
test QWK=0.751
test Acc=0.757
test F1=0.757

test/certo MAE=0.155
test/certo QWK=0.845
test/certo Acc=0.851
test/certo F1=0.840
```

Matriz de confusão:

```text
             Pred Baixo  Pred Médio  Pred Alto
True Baixo       77          32          1
True Médio       12         106         11
True Alto         0          17         44
```

Leitura:

- v2 melhorou todas as métricas principais.
- Classe `Médio` ficou mais aprendível:
  - acertos de `Médio`: `88 -> 106`
  - `Médio -> Baixo`: `20 -> 12`
  - `Médio -> Alto`: `22 -> 11`
- Tradeoff: o modelo ficou um pouco mais conservador para extremos,
  especialmente `Alto -> Médio`.

## Exp 03 - Dataset v2 Com Horizontal Flip

Hipótese:

```text
Grass on the left e grass on the right deveriam ser semanticamente equivalentes.
HorizontalFlip pode reduzir viés de lado e aumentar diversidade sem coletar
novos dados.
```

Risco:

```text
As imagens vêm de câmera fisheye/rodovia com geometria consistente. O flip pode
remover viés útil ou criar geometria pouco realista. Por isso deve ser uma
ablation, não default.
```

Comando recomendado:

```bash
python training/train.py \
  --annotations MapillaryDatasetL+R-Annoted-v2.json \
  --batch-size 16 \
  --grad-accum-steps 1 \
  --hflip-p 0.5 \
  --run-name v2-hflip-p05-seed42
```

Critério de decisão:

```text
Manter HorizontalFlip se melhorar QWK/MAE no teste sem piorar muito os extremos.
Olhar especialmente:
- test QWK
- test MAE
- test/certo QWK
- Alto -> Médio
- Baixo -> Médio
- erros de 2 classes
```

Resultado:

```text
Run: training/runs/v2-hflip-p05-seed42

Melhor validacao:
finetune epoch 13
val MAE=0.221
val QWK=0.761
val Acc=0.789
val F1=0.777

val/certo MAE=0.155
val/certo QWK=0.838
val/certo Acc=0.851
val/certo F1=0.838

Teste:
test MAE=0.263
test QWK=0.731
test Acc=0.743
test F1=0.741

test/certo MAE=0.144
test/certo QWK=0.866
test/certo Acc=0.856
test/certo F1=0.846
```

Matriz de confusão:

```text
             Pred Baixo  Pred Médio  Pred Alto
True Baixo       79          30          1
True Médio       14         102         13
True Alto         1          18         42
```

Comparação contra Exp 02 sem flip:

```text
Sem flip:
test MAE=0.247 | QWK=0.751 | Acc=0.757 | F1=0.757
test/certo MAE=0.155 | QWK=0.845 | Acc=0.851 | F1=0.840

Com flip p=0.5:
test MAE=0.263 | QWK=0.731 | Acc=0.743 | F1=0.741
test/certo MAE=0.144 | QWK=0.866 | Acc=0.856 | F1=0.846
```

Leitura:

- `HorizontalFlip(p=0.5)` melhorou o subconjunto `certo`.
- No conjunto completo, piorou MAE, QWK, accuracy e F1.
- A piora apareceu principalmente em `Médio` e `Alto`:
  - `Médio` correto: `106 -> 102`
  - `Médio -> Alto`: `11 -> 13`
  - `Alto` correto: `44 -> 42`
  - apareceu `Alto -> Baixo`: `0 -> 1`
- Decisão atual: não adotar `p=0.5` como default. Manter como experimento
  opcional; se testar novamente, usar `p=0.25` ou comparar múltiplas seeds.
