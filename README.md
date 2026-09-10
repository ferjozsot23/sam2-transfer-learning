# Segmentación semántica de SuperTuxKart con SAM 2

Transfer learning desde **Segment Anything Model 2** para clasificar cada píxel de un
frame del videojuego en una de 7 clases: `background`, `track`, `kart`, `pickup`,
`nitro`, `bomb`, `projectile`.

*Proyecto Final · Visión Artificial 202610 · Universidad San Francisco de Quito*

**mIoU 0.5320** sobre dos circuitos nunca vistos, entrenando el **3.6% del modelo**.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figuras/cualitativo-dark.png">
  <img src="docs/figuras/cualitativo-light.png" alt="Predicciones sobre circuitos de validación">
</picture>

Ninguno de estos circuitos se usó para entrenar.

---

## Por qué SAM 2 no sirve tal cual

SAM 2 **no es un segmentador semántico**. Es *promptable* y *binario*: recibe una imagen
más un prompt —un punto, una caja— y devuelve la máscara del objeto señalado. No sabe qué
es un kart ni distingue siete clases.

Lo que sí tiene es un **image encoder** (Hiera + cuello FPN) entrenado sobre millones de
máscaras. El transfer learning consiste en:

1. Usar ese encoder **congelado** como extractor de features.
2. Descartar la *memory attention* (es para vídeo) y el *mask decoder* original (es binario).
3. Entrenar **una cabeza nueva** que fusione los tres niveles del FPN y produzca 7 logits
   por píxel.

```
frame (B,3,448,448)
      ↓
Image encoder de SAM 2  ← CONGELADO, 69.1 M parámetros
      ↓
backbone_fpn: 3 niveles de 256 canales, strides 4 / 8 / 16
      ↓
Cabeza FPN  ← ENTRENABLE, 2.56 M
      ↓
logits (B,7,448,448) → interpolados al tamaño original
```

---

## Probarlo

```bash
pip install -r requirements.txt
python predict.py ejemplos/
python predict.py mis_imagenes/ --out salida/ --masks
```

Cada imagen produce un panel `original | segmentación | superposición`. Con `--masks`
guarda además la máscara cruda de 1 canal con valores 0–6.

### Desde tu propio código

```python
from models import load_model

model = load_model('model.th')
pred  = model(x).argmax(1)
```

`x` es un tensor `(B,3,H,W)` en `[0,1]` y la salida `(B,H,W)` con el índice de clase de
cada píxel. Cualquier resolución vale — verificado desde 211×333 hasta 1080×1920.

`model.th` pesa **10 MB** porque guarda solo la cabeza entrenada; el encoder de SAM 2 se
reconstruye al cargar, descargando su checkpoint de HuggingFace la primera vez.

---

## Resultados

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figuras/iou-dark.png">
  <img src="docs/figuras/iou-light.png" alt="IoU por clase, SAM 2 frente a U-Net">
</picture>

| clase | IoU |
|---|---|
| background | 0.8595 |
| track | 0.8572 |
| kart | 0.7744 |
| pickup | 0.6336 |
| nitro | 0.4614 |
| bomb | 0.1377 |
| projectile | 0.0000 |
| **mIoU** | **0.5320** |

Medido con una matriz de confusión acumulada sobre los 500 frames de validación, nunca
promediando IoU por lote.

**El split es por circuito, no por frame.** Los frames de un mismo circuito son casi
idénticos —el kart avanza unos centímetros por frame— así que un split aleatorio pondría
frames gemelos a ambos lados y daría un mIoU alto y falso.

| | circuitos | frames |
|---|---|---|
| entrenamiento | `abyss`, `gran_paradiso_island`, `hacienda`, `olivermath` | 1000 |
| validación | `lighthouse`, `volcano_island` | 500 |

### Contra una U-Net entrenada desde cero

El mismo dataset y el mismo split se resolvieron antes sin transfer learning:

| clase | U-Net | SAM 2 | |
|---|---|---|---|
| background | 0.7768 | **0.8595** | +0.083 |
| track | 0.7365 | **0.8572** | +0.121 |
| kart | 0.6709 | **0.7744** | +0.104 |
| pickup | 0.5013 | **0.6336** | +0.132 |
| bomb | 0.0194 | **0.1377** | +0.118 |
| nitro | **0.5352** | 0.4614 | −0.074 |
| **mIoU** | 0.4629 | **0.5320** | **+0.069** |

Un **15% relativo**, ganando en 5 de 7 clases y entrenando el 3.6% de los parámetros. El
error dominante de la U-Net era confundir carretera con paisaje por brillo y textura en
circuitos no vistos; `track` pasa de 0.7365 a 0.8572.

---

## Qué mueve la aguja y qué no

Se probaron **33 configuraciones** (`experimentos/resultados/`). Solo los ejes de
capacidad tienen efecto real:

| eje | efecto |
|---|---|
| backbone `small` → `large` | **+0.020** |
| `dim` de la cabeza 128 → 256 | **+0.020** |
| `dim` 256 → 512 | −0.012, se agota |
| resolución 448 → 1024 | +0.001 por 5–7× el coste |
| learning rate 1e-3 ↔ 3e-4 | 0.001, nada |
| exponente de pesos 0.20 ↔ 0.30 | 0.000, nada |

**Descongelar el encoder empeora.** Se probó liberar el cuello FPN (0.3% del encoder) con
learning rate diez veces menor, midiendo el congelado como control en el mismo barrido:
**−0.017 de mIoU y el doble de inestabilidad**. Con 1000 imágenes de 4 circuitos, adaptar
los pesos preentrenados solo añade sobreajuste. El código lo soporta
(`--modos none neck bN`, `--encoder-lr`) y queda como trabajo futuro reproducible.

**Sobre la fiabilidad.** La σ del mIoU entre épocas consecutivas es **0.018**, así que dos
configuraciones que difieran menos de ~0.036 son indistinguibles. Las decisiones se tomaron
con medias por eje (8–12 corridas, error típico ~0.006), no con el ranking individual. El
0.5320 es el máximo de ~800 evaluaciones y está optimistamente sesgado; el nivel esperado
de esta configuración es **~0.518**, confirmado por dos corridas independientes.

---

## Limitaciones

**`projectile` aparece en 1 frame de los 1000** de entrenamiento y `bomb` en 168. Ninguna
arquitectura aprende una clase que casi no existe. Aun así, una configuración descartada
llegó a **0.2422 de IoU en `projectile`** partiendo de ese único ejemplo — algo que la
U-Net nunca logró en 22 configuraciones.

---

## Reproducir

```bash
python train.py --data-root data --backbone base_plus --size 448 --dim 256 \
    --lr 3e-4 --weight-power 0.20 --epochs 40 --batch-size 8

python experimentos/sweep.py --data-root data --backbones small base_plus large \
    --dims 128 256 --lrs 1e-3 3e-4 --powers 0.20 0.30
```

El dataset se organiza en `images/` y `masks/`, emparejados por circuito e identificador.
La máscara se lee sin conversión de color y se redimensiona con **NEAREST**: un
`ToTensor()` la dividiría entre 255 y destruiría la codificación de clases, e interpolarla
inventaría clases inexistentes en los bordes.

```
train.ipynb           ENTREGABLE — métricas e imágenes segmentadas
utils.py              ENTREGABLE — dataset, métricas, pesos de clase
model.th              ENTREGABLE — modelo entrenado (10 MB)
models.py             la U-Net sobre SAM 2, save_model / load_model
train.py              loop de entrenamiento, --resume, descongelado parcial
predict.py            inferencia sobre imagen o carpeta
class_weights.json    conteo de píxeles del conjunto de entrenamiento
ejemplos/             imágenes para probar sin descargar el dataset
experimentos/         barrido multi-GPU y las 33 configuraciones probadas
docs/                 figuras del README y el script que las regenera
servidor/             orquestación del entrenamiento en el DGX
```

Entrenado en un DGX H200 dentro de contenedor Docker (torch 2.13, CUDA); el `.th`
verificado en macOS con torch 2.14 sobre CPU. Depende de `torch`, `torchvision`, `numpy`,
`Pillow` y `sam2`.
