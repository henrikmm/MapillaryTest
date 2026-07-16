# Mapillary probe: sequence Vzro8mDBsivZkEFnxG9utg

## Resultado

- Imagens na sequência: `268`
- Imagens baixadas: `268`
- Pasta das imagens: `images/`
- Resolução baixada: `thumb_original_url`
- Dimensão validada em amostra: `4624x3468`

## Consistência dos intrínsecos

Esta sequência é mais consistente para FOV do que `by41kl5i2vQmCTx0NOsqfV`.

- `camera_type`: `perspective` em `268/268`
- `make`: `samsung`
- `model`: `SM-M315F`
- `camera_parameters`: presente em `268/268`
- conjuntos distintos de `camera_parameters`: `47`

Embora existam mais conjuntos numéricos distintos do que na sequência anterior, a faixa de FOV estimada é bem mais estreita:

```text
horizontal_fov_deg: min 67.61, max 76.36, média 72.85
vertical_fov_deg:   min 53.33, max 61.07, média 57.93
diagonal_fov_deg:   min 79.86, max 89.02, média 85.37
```

Comparação com a sequência anterior:

```text
by41kl5i2vQmCTx0NOsqfV horizontal_fov_deg: min 60.93, max 121.17, média 82.09
Vzro8mDBsivZkEFnxG9utg horizontal_fov_deg: min 67.61, max 76.36, média 72.85
```

Portanto, para um pipeline que precisa escolher uma calibração/FOV aproximado da sequência, `Vzro8mDBsivZkEFnxG9utg` é a opção melhor.

## Arquivos

- `metadata_all.jsonl`: metadados de todos os frames.
- `summary.json`: resumo agregado e status do download.
- `field_probe_first_image.json`: campos testados na API.
- `sequence_image_ids.json`: lista dos IDs da sequência.
- `images/`: imagens baixadas.

