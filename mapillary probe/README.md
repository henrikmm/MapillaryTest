# Mapillary probe: sequence by41kl5i2vQmCTx0NOsqfV

Consulta feita em `https://graph.mapillary.com` usando o token local do projeto.

## Arquivos

- `probe_mapillary_sequence.py`: script usado para consultar a API.
- `sequence_image_ids.json`: IDs das imagens retornadas pela sequência.
- `field_probe_first_image.json`: teste campo a campo no primeiro frame.
- `metadata_all.jsonl`: metadados completos retornados para todos os frames, um JSON por linha.
- `sample_metadata_first_image.json`: primeiro frame expandido.
- `summary.json`: resumo agregado.

## Resultado principal

- Total de imagens na sequência: `122`
- `camera_type`: `perspective` em `122/122`
- `make`: `Blackvue2`
- `model`: `DR900S-2CH`
- resolução original: `3840x2160`
- `camera_parameters`: presente em `122/122`

Na API, `camera_parameters` vem como:

```text
[focal, k1, k2]
```

Para esta sequência houve `9` conjuntos distintos de parâmetros. Os mais comuns foram:

```text
41 frames: [0.85, 0, 0]
40 frames: [0.281896086661, -0.072679555268, 0.003518344952]
10 frames: [0.833058287187, -0.006821474381, -0.002092026636]
9 frames:  [0.77403011988, -0.005695514333, -0.002929007018]
7 frames:  [0.702988627929, -0.007951395407, -0.003847600527]
6 frames:  [0.756469885778, -0.003769398556, -0.001089202078]
4 frames:  [0.847404676717, -0.001201396865, -0.000574543628]
3 frames:  [0.793651388194, -0.001886370525, -0.001740956614]
2 frames:  [0.835622, 0.000139, 0.000426]
```

## Estimativa de FOV

A API não retorna um campo `fov` direto. O script calcula uma estimativa assumindo a convenção do OpenSfM: coordenadas normalizadas pela maior dimensão da imagem. Para uma imagem `3840x2160`, usei:

```text
fx_px = focal * max(width, height)
horizontal_fov = 2 * atan(width / (2 * fx_px))
vertical_fov = 2 * atan(height / (2 * fx_px))
diagonal_fov = 2 * atan(hypot(width, height) / (2 * fx_px))
```

Resumo das estimativas, tratando a câmera como pinhole:

```text
horizontal_fov_deg: min 60.93, max 121.17, média 82.09
vertical_fov_deg:   min 36.62, max 89.87,  média 55.06
diagonal_fov_deg:   min 68.03, max 127.66, média 89.06
```

Exemplo do primeiro frame (`141823754696982`):

```json
{
  "camera_type": "perspective",
  "width": 3840,
  "height": 2160,
  "camera_parameters": [0.75646988577834, -0.0037693985563277, -0.0010892020782009],
  "horizontal_fov_deg_assuming_pinhole": 66.92660337215816,
  "vertical_fov_deg_assuming_pinhole": 40.789536006687726,
  "diagonal_fov_deg_assuming_pinhole": 74.35018425960935
}
```

## Campos acessíveis testados

Todos estes campos foram aceitos pela API. Quase todos vieram em `122/122`; `mesh` veio em `81/122`.

```text
id
altitude
atomic_scale
camera_parameters
camera_type
captured_at
compass_angle
computed_altitude
computed_compass_angle
computed_geometry
computed_rotation
creator
exif_orientation
geometry
height
make
model
thumb_256_url
thumb_1024_url
thumb_2048_url
thumb_original_url
merge_cc
mesh
quality_score
sequence
sfm_cluster
width
```

