# Plano de Treinamento — Classificador de Altura de Grama

## 1. Objetivo
Treinar um classificador ordinal para prever a altura visual da grama lateral
em 3 classes:

- `Baixo` = 0
- `Médio` = 1
- `Alto` = 2

O problema é **ordinal**, não nominal: errar `Baixo -> Médio` é menos grave do
que errar `Baixo -> Alto`.

---

## 2. O Que Foi Anotado de Verdade

### Unidade de anotação
- A anotação foi feita **somente nos crops**.
- Cada amostra anotada é um crop em:
  - `crops/left/{image_id}.jpg`
  - `crops/right/{image_id}.jpg`
- O ground truth é o arquivo `MapillaryDatasetL+R-Annoted.json`.

### O que NÃO foi a entrada do anotador
- `helpers/segmented/` não foi a base real de anotação do dataset final.
- `helpers/masks/` não foi anotado manualmente; serviu apenas para gerar crops.
- `helpers/explore_crop_features.py` foi só exploração auxiliar e não participou
  do processo de anotação.

### Consequência prática
O modelo principal deve ser pensado primeiro como um **classificador de crops**.
Se quisermos usar mais contexto espacial no futuro, isso vira um experimento ou
ablation, não a definição do ground truth.

---

## 3. Estrutura Atual do Projeto

```text
MapillaryTest/
├── images/                              ← frames originais
├── crops/
│   ├── left/                            ← crops anotados
│   └── right/                           ← crops anotados
├── helpers/
│   ├── masks/                           ← máscaras terrain-only
│   ├── segmented/                       ← visualizações auxiliares
│   ├── explore_crop_features.py         ← exploração opcional
│   └── exploratory_crop_features.csv
├── label_studio/
│   ├── generate_crops.py                ← fluxo real de geração dos crops
│   └── template.xml
├── training/
│   ├── TRAINING_PLAN.md
│   └── smoke_test.py
└── MapillaryDatasetL+R-Annoted.json
```

### Core para treino
- `crops/left/*.jpg`
- `crops/right/*.jpg`
- `MapillaryDatasetL+R-Annoted.json`

### Auxiliar
- `helpers/masks/`
- `helpers/segmented/`
- `helpers/explore_crop_features.py`
- `helpers/exploratory_crop_features.csv`

---

## 4. Resumo do Dataset

### Volume
- 999 imagens originais fisheye, 2704×2028 px
- 1987 crops anotados
- 997 `image_id`s com pelo menos um lado anotado
- 990 imagens com os dois lados anotados

### Distribuição de classes
```text
Baixo (0): 705   (35.5%)
Médio (1): 896   (45.1%)
Alto  (2): 386   (19.4%)
```

### Confiança
```text
certo   : 1122 (56.5%)
incerto :  865 (43.5%)
```

### Observações importantes
- O lado `right` tem bem mais `Alto` do que o `left`.
- Isso é uma característica do dataset, não um motivo para passar `side` como
  feature explícita.
- Há uma cauda de crops com área de grama muito pequena; isso pode introduzir
  exemplos de baixa evidência, principalmente na classe `Baixo`.

---

## 5. Decisões Consolidadas Para o Baseline

Este é o baseline que deve ser implementado primeiro.

```text
Entrada:
- crops RGB atuais em crops/left e crops/right
- sem side como feature explícita
- sem máscara como canal extra no baseline
- sem features manuais no baseline

Preprocessing:
- resize preservando aspect ratio
- padding preto até 384x384
- normalização ImageNet

Modelo:
- EfficientNet-B2 via timm
- pesos pretrained ImageNet
- num_classes=2 para CORAL com 3 classes ordinais

Loss:
- CoralLoss
- peso por confiança:
  - certo = 1.0
  - incerto = 0.6

Split:
- por image_id
- 70% treino, 15% validação, 15% teste

Métricas principais:
- MAE ordinal
- QWK

Hardware alvo:
- RTX 4060 Ti 8 GB
- batch_size = 8
- gradient_accumulation = 2
- batch efetivo = 16
- AMP/mixed precision ligado
```

Racional: antes de adicionar máscaras extras, features manuais ou arquiteturas
mais complexas, o primeiro experimento precisa medir se os crops anotados já
contêm sinal suficiente para a tarefa ordinal.

---

## 6. Qual Framework Escolher

## Recomendação
Usar **PyTorch** para o treino do modelo, com:
- `timm` para o backbone
- `coral_pytorch` para o ordinal regression
- `albumentations` para preprocessing/augmentation
- `scikit-learn` para splits e métricas

## PyTorch
### Vantagens
- Melhor encaixe com `timm` e `coral_pytorch`
- Mais natural para freeze/unfreeze, LR diferencial e losses customizadas
- Mais fácil para loops de treino experimentais e ablations
- Mais transparente para depurar

### Desvantagens
- Mais código manual
- Exige mais disciplina na implementação

## Keras / TensorFlow
### Vantagens
- API mais alta e mais rápida para protótipos simples
- Treino padronizado e callbacks prontos

### Desvantagens
- Menos natural para este stack específico
- Menos alinhado com `timm` e com a implementação pronta de CORAL que você já
  validou
- Para custom ordinal logic, costuma ficar menos direto do que em PyTorch

## scikit-learn
### Onde entra bem
- `StratifiedGroupKFold`
- `GroupShuffleSplit`
- `cohen_kappa_score`
- análise de resultados
- possíveis baselines tabulares com features manuais

### Onde NÃO entra bem
- treino end-to-end de EfficientNet em imagens

## Decisão prática
- **Treino do CNN:** PyTorch
- **Splits e métricas:** scikit-learn
- **Keras:** só valeria a pena se o objetivo fosse simplicidade máxima e se você
  abrisse mão do stack atual

---

## 7. Modelo Recomendado

## Backbone
`EfficientNet-B2` via `timm`

### Por que faz sentido aqui
- Modelo relativamente leve
- Bom histórico em transfer learning com datasets moderados
- Mais simples de treinar e estudar do que uma arquitetura maior
- ConvNet ainda casa bem com padrões de textura, densidade e volume visual da
  grama

### Tradeoffs
- Menos poderoso do que backbones maiores ou transformers modernos
- Se a tarefa depender muito de contexto geométrico amplo, o crop apertado pode
  limitar o potencial do backbone

---

## 8. O Que é CORAL e Por Que Usar

## Resposta curta
Sim: no uso prático, `CORAL` é o **framework ordinal** e `CoralLoss` é a loss
usada para treinar esse framework.

## Intuição
Em vez de prever 3 classes independentes com softmax, o modelo aprende
`K - 1` decisões ordenadas.

Para 3 classes:
- saída 1: `é maior que Baixo?`
- saída 2: `é maior que Médio?`

Encoding:
```text
Baixo -> [0, 0]
Médio -> [1, 0]
Alto  -> [1, 1]
```

Decoding:
- `[0, 0]` => `Baixo`
- `[1, 0]` => `Médio`
- `[1, 1]` => `Alto`

## Por que isso ajuda
- Respeita a ordem natural das classes
- Penaliza implicitamente saltos grandes
- Evita tratar `Baixo`, `Médio` e `Alto` como categorias sem relação

## Por que CORAL em vez de softmax
- Softmax clássico usa cross-entropy multiclasse e não modela a ordem
- CORAL impõe uma estrutura ordinal consistente
- Para 3 classes ambíguas e fronteiras suaves, isso costuma ser uma escolha bem
  alinhada ao problema

## Limitação importante
CORAL é ótimo como baseline ordinal, mas não resolve sozinho:
- ruído de rótulo
- ambiguidade `certo/incerto`
- perda de contexto causada pelos crops

## CORN
`CORN` é uma alternativa da mesma família ordinal.
Vale como ablation depois, porque às vezes performa melhor que CORAL em certos
datasets e arquiteturas.

---

## 9. Loss, Erro, Backpropagation e Métricas

## O modelo usa backpropagation?
Sim.

O fluxo continua sendo o padrão:
1. forward pass
2. cálculo da loss
3. backpropagation
4. update dos pesos com optimizer

Nada muda nisso. O que muda é **qual loss** estamos otimizando.

## O que é a loss
A loss é a função que o treinamento tenta minimizar.

No plano principal:
- a loss é **`CoralLoss`**

Ela é a medida de erro usada no treino.

## E MSE? E entropy?

### MSE
- `MSE` = mean squared error
- é muito comum em regressão
- pode ser usado como métrica ordinal se o alvo for tratado como número
- não é a loss principal recomendada aqui

### Cross-entropy
- é a loss padrão de classificação multiclasse com softmax
- boa quando as classes são independentes
- aqui perde o benefício da ordinalidade

### CoralLoss
- é a loss principal recomendada
- foi desenhada para a formulação ordinal CORAL

## Loss versus métrica
- **loss**: usada para treinar
- **métrica**: usada para interpretar a qualidade do modelo

Um modelo pode treinar com `CoralLoss` e ser avaliado com `MAE`, `QWK`,
`accuracy` e `F1`.

---

## 10. Métricas Recomendadas

## Principais
- **MAE ordinal**
- **QWK** (`quadratic weighted kappa`)

## Suplementares
- Accuracy
- F1 macro
- Confusion matrix

## Por que MAE ordinal
MAE em classes ordinais mede quantos degraus, em média, o modelo erra.

Exemplo:
- prever `Médio` quando o certo era `Baixo` => erro `1`
- prever `Alto` quando o certo era `Baixo` => erro `2`

Isso combina muito bem com a tarefa.

## Por que QWK
QWK mede concordância penalizando mais fortemente erros distantes.
É uma métrica clássica para problemas ordinais e costuma ser mais informativa do
que accuracy pura.

## Por que accuracy sozinha não basta
Accuracy trata qualquer erro como igual.
Para este problema, isso é ruim.

Exemplo:
- `Baixo -> Médio` e `Baixo -> Alto` contam igual na accuracy
- mas operacionalmente não são iguais

## Meta inicial razoável
- MAE ordinal baixo
- QWK claramente acima do acaso

Eu manteria:
- **métrica principal de leitura:** `MAE + QWK`
- **métricas auxiliares:** `accuracy + F1 macro`

---

## 11. Resolução das Imagens e Impacto no Modelo

## A preocupação é válida
Os crops usados na anotação têm resolução alta, mas o modelo vai receber uma
versão muito menor.

Isso afeta:
- textura fina
- detalhes pequenos das folhas
- microestruturas da vegetação

## O que se perde
- detalhe local fino
- pequenas pistas de textura

## O que ainda pode sobreviver
- volume visual
- densidade aparente
- ocupação da lateral
- relação entre massa de grama e acostamento

## Problema extra do plano antigo
Redimensionar direto para `288x288` pode deformar crops muito alongados.
Essa deformação geométrica é pior para esta tarefa do que simplesmente reduzir a
resolução.

## Recomendação prática
**Não distorcer o aspect ratio.**

Melhor pipeline:
1. redimensionar preservando proporção
2. fazer padding preto até tamanho fixo
3. normalizar

Exemplo de alvo:
- `320x320` ou `384x384` com padding preto

## Tradeoff de resolução

### Menor resolução
Vantagens:
- treino mais rápido
- batch maior
- menos memória
- menos risco de overfit

Desvantagens:
- perde detalhe fino

### Maior resolução
Vantagens:
- preserva mais textura e bordas

Desvantagens:
- mais custo
- batch menor
- mais instabilidade em dataset pequeno

## Recomendação
Começar com:
- `384x384` com aspect ratio preservado + padding

Depois do baseline, comparar com:
- `320x320` com aspect ratio preservado + padding

Eu não começaria com resize deformando para quadrado puro. No teste visual
feito nos crops, `384x384` preservou melhor textura e volume sem custo excessivo
para a RTX 4060 Ti 8 GB quando usado com AMP e batch pequeno.

---

## 12. Preprocessing e Augmentation

## Diretriz principal
Evitar transformações que alterem a geometria da cena.

## O que evitar
- `RandomResizedCrop`
- rotações arbitrárias
- crops aleatórios agressivos
- warps que mudem escala aparente da vegetação

## O que pode usar
- jitter de brilho
- jitter de contraste
- jitter moderado de saturação
- leve blur
- leve ruído
- pequenas variações fotométricas

## Pipeline sugerido

### Treino
```python
albumentations.Compose([
    A.LongestMaxSize(max_size=384, interpolation=cv2.INTER_CUBIC),
    A.PadIfNeeded(
        min_height=384,
        min_width=384,
        border_mode=cv2.BORDER_CONSTANT,
        fill=(0, 0, 0),
    ),
    A.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.15, hue=0.02, p=0.7),
    A.GaussNoise(p=0.15),
    A.GaussianBlur(blur_limit=(3, 5), p=0.15),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
```

### Validação / teste
```python
albumentations.Compose([
    A.LongestMaxSize(max_size=384, interpolation=cv2.INTER_CUBIC),
    A.PadIfNeeded(
        min_height=384,
        min_width=384,
        border_mode=cv2.BORDER_CONSTANT,
        fill=(0, 0, 0),
    ),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
```

---

## 13. Como Fine-Tunar a EfficientNet para Este Problema

## Etapa 1 — criar o modelo
```python
model = timm.create_model(
    "efficientnet_b2",
    pretrained=True,
    num_classes=2,   # K-1 saídas para CORAL com K=3 classes
    drop_rate=0.3,
)
```

## Etapa 2 — treinar a head primeiro
Congelar backbone e treinar só a camada final.

### Por que fazer isso
- estabiliza o começo do treino
- evita destruir os pesos pré-treinados cedo demais
- ajuda quando o dataset é relativamente pequeno

### Sugestão
- epochs: `5`
- LR head: `1e-3`

## Etapa 3 — unfreeze parcial
Descongelar os últimos blocos e continuar o fine-tuning.

### Por que
- as camadas mais profundas precisam se adaptar ao domínio da grama
- as primeiras camadas geralmente já capturam bordas e texturas úteis

### Sugestão
- unfreeze dos últimos blocos
- LR backbone: `3e-5`
- LR head: `3e-4`
- epochs: `25`

## Etapa 4 — opcional full unfreeze
Só se o treino estiver estável e houver ganho claro.

### Risco
- overfit
- forgetting do pré-treino
- instabilidade maior

## Optimizer
- `AdamW`

## Scheduler
- warmup curto
- depois cosine decay

## Best practices
- salvar melhor checkpoint por `QWK` ou `MAE`
- usar early stopping
- monitorar train vs val para detectar overfit
- manter seeds fixas
- registrar config de cada experimento

## O que evitar
- LR muito alto após unfreeze
- treinar tudo de uma vez desde a epoch 1
- usar augmentations geométricas fortes
- comparar experimentos com splits diferentes

## Tradeoffs

### Freeze parcial
Vantagem:
- mais estável

Desvantagem:
- pode limitar adaptação ao domínio

### Full fine-tuning
Vantagem:
- potencialmente melhor performance

Desvantagem:
- mais risco de overfit e instabilidade

---

## 14. Pipeline Otimizado Para RTX 4060 Ti 8 GB

Configuração inicial recomendada:

```text
image_size: 384
batch_size: 8
grad_accum_steps: 2
effective_batch_size: 16
num_workers: 4
pin_memory: true
AMP: true
TF32: true
```

Se couber com folga:

```text
batch_size: 16
grad_accum_steps: 1
```

Se der `CUDA out of memory`:

```text
batch_size: 4
grad_accum_steps: 4
```

Boas práticas adotadas no treino:
- manter imagens e transforms na CPU
- enviar para GPU somente o batch
- usar `optimizer.zero_grad(set_to_none=True)`
- usar mixed precision apenas em CUDA
- salvar checkpoints fora do git em `training/runs/`

---

## 15. Split: Holdout Simples ou Validação Agrupada

## Regra obrigatória
Split por `image_id`, nunca por crop.

Left e right da mesma imagem precisam ir para o mesmo split.

## Opção A — single holdout
Exemplo:
- treino
- validação
- teste

### Vantagens
- simples
- rápido
- fácil de implementar
- ótimo para começar

### Desvantagens
- alta variância
- resultado pode depender demais do split escolhido

## Opção B — grouped cross-validation
Exemplo:
- `StratifiedGroupKFold`

### Vantagens
- estimativa mais robusta
- menos dependente de um split específico
- melhor para comparar configs

### Desvantagens
- custo computacional muito maior
- exige mais organização experimental

## Recomendação prática
Fase de desenvolvimento:
- usar **single holdout grouped 70/15/15**

Fase de confirmação:
- rodar os melhores modelos em **grouped CV**

Assim você não paga o custo total logo no começo, mas também não toma decisão
final com base em um único split.

---

## 16. Como Tratar `certo` e `incerto`

## O que significa hoje
- `certo` = caso relativamente limpo
- `incerto` = fronteira ambígua ou caso de baixa confiança

## Opção 1 — tratar tudo igual
### Vantagem
- simples

### Desvantagem
- injeta mais ruído no treino

## Opção 2 — treinar com pesos por amostra
Exemplo:
- `certo` = peso `1.0`
- `incerto` = peso `0.5` ou `0.7`

### Vantagens
- preserva dados
- reduz impacto dos casos ambíguos
- fácil de implementar

### Desvantagens
- precisa escolher o peso

## Opção 3 — soft labels entre classes adjacentes
Exemplo:
- `Médio incerto` poderia virar algo como distribuição entre `Baixo/Médio` ou
  `Médio/Alto`, dependendo da regra

### Vantagens
- modela melhor a ambiguidade

### Desvantagens
- mais difícil de definir corretamente
- aumenta complexidade do treino

## Recomendação
Começar com:
- treino com todos os dados
- **peso menor para `incerto`: 0.6**
- avaliação reportada em:
  - conjunto completo
  - subconjunto `certo`

Essa é, na prática, a escolha mais equilibrada para um baseline sério.

---

## 17. Smoke Test e Ambiente

O arquivo `training/smoke_test.py` continua útil, mas ele deve ser executado no
ambiente de treino correto.

Observação importante:
- este workspace pode não ter todas as dependências ativas no `python3` atual
- então o smoke test deve ser tratado como verificação do **ambiente de treino**,
  não como garantia universal do shell atual

---

## 18. Experimentos Recomendados

## Prioridade alta
1. Baseline crop-only com EfficientNet-B2 + CORAL
2. Comparação de resolução:
   - `384x384` com padding
   - `320x320` com padding, depois do baseline
3. `incerto` com peso reduzido

## Prioridade média
4. HorizontalFlip `p=0.5` como ablation no dataset v2
5. CORN vs CORAL
6. grouped CV para os melhores configs

## Prioridade baixa
7. fusão com features exploratórias
8. uso explícito de máscara como canal extra

Hoje, eu colocaria **features exploratórias** e **mask channel** abaixo de
resolver bem:
- aspect ratio
- split grouped
- tratamento de `incerto`
- consistência da classe `Médio`

---

## 19. Stack Recomendado

```text
Treino principal:
- PyTorch
- timm
- coral_pytorch
- albumentations

Suporte:
- scikit-learn
- numpy
- pandas
- matplotlib / seaborn
```

---

## 20. Resumo Executivo

- O dataset anotado final é **crop-based**.
- `helpers/masks/` e `helpers/segmented/` são auxiliares, não entrada core do
  treino.
- A escolha mais coerente hoje é:
  - **PyTorch + timm + coral_pytorch**
- `CORAL` faz sentido porque a tarefa é ordinal.
- O modelo continua treinando com **backpropagation normal**.
- A loss principal recomendada é **`CoralLoss`**.
- As melhores métricas para leitura do problema são **MAE ordinal + QWK**.
- Evitar augmentations geométricas fortes é a decisão correta.
- O próximo cuidado importante é **preservar aspect ratio** ao reduzir a
  resolução; o baseline consolidado usa **384x384 com padding**.
- Para `incerto`, a melhor primeira abordagem é **usar todos os dados com peso
  0.6**, em vez de descartar ou tratar igual sem reflexão.
