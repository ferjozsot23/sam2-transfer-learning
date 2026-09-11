# Segmentación semántica de SuperTuxKart con SAM 2

Transfer learning desde **Segment Anything Model 2** para clasificar cada píxel de un
frame del videojuego en una de 7 clases: `background`, `track`, `kart`, `pickup`,
`nitro`, `bomb`, `projectile`.

**mIoU 0.6008** en dos circuitos que no se usaron para entrenar.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figuras/cualitativo-dark.png">
  <img src="docs/figuras/cualitativo-light.png" alt="Predicciones sobre circuitos de validación">
</picture>

---

## Enfoque

SAM 2 **no es un segmentador semántico**. Es *promptable* y *binario*: recibe una imagen
más un prompt —un punto, una caja— y devuelve la máscara del objeto señalado. No sabe qué
es un kart ni distingue siete clases.

Lo que sí tiene es un **image encoder** entrenado sobre millones de máscaras. El modelo:

1. Reutiliza ese encoder, con sus pesos preentrenados, como extractor de features.
2. Descarta la *memory attention* (es para vídeo) y el *mask decoder* original (es binario).
3. Añade una **cabeza nueva** que fusiona los tres niveles del FPN y produce 7 logits por píxel.
4. **Afina el encoder entero** con un learning rate cien veces menor que el de la cabeza.

## Arquitectura

- **Encoder**: SAM 2.1 Hiera-Large, pesos preentrenados, afinado entero con `lr` 1e-5
  (212.7 M parámetros).
- **Features**: `backbone_fpn`, 3 niveles de 256 canales a strides 4, 8 y 16.
- **Cabeza**: FPN de grueso a fino con `lr` 1e-3 (2.56 M parámetros), terminada en una
  convolución 1×1 a 7 logits por píxel, sin softmax.
- **Entrada y salida**: el `forward` normaliza, reescala a 448×448 y devuelve los logits al
  tamaño original de la imagen, así que acepta cualquier resolución.

---

## Probarlo

### Sin instalar nada — Colab

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ferjozsot23/sam2-transfer-learning/blob/main/demo.ipynb)

Abre el notebook, *Ejecutar todas*, y sube tus imágenes cuando lo pida. Descarga el modelo
la primera vez y devuelve las máscaras en un zip.

### En local

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
cada píxel.

**`model.th` pesa 861 MB** y está en *Releases*; `predict.py` y el notebook lo descargan
solos. Incluye el encoder afinado completo, así que carga sin conexión.

---

## Resultados

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figuras/iou-dark.png">
  <img src="docs/figuras/iou-light.png" alt="IoU por clase, SAM 2 frente a U-Net">
</picture>

| clase | IoU |
|---|---|
| background | 0.8791 |
| track | 0.8743 |
| kart | 0.8158 |
| pickup | 0.5496 |
| nitro | 0.4562 |
| bomb | 0.2180 |
| projectile | 0.4124 |
| **mIoU** | **0.6008** |

Medido con una matriz de confusión acumulada sobre los 500 frames de validación. Es la
mejor de 4 repeticiones de la misma configuración; la media de las cuatro es 0.571.

**El split es por circuito, no por frame.** Los frames de un mismo circuito son casi
idénticos, así que un split aleatorio pondría frames gemelos a ambos lados y daría un mIoU
alto y falso.

| | circuitos | frames |
|---|---|---|
| entrenamiento | `abyss`, `gran_paradiso_island`, `hacienda`, `olivermath` | 1000 |
| validación | `lighthouse`, `volcano_island` | 500 |

### Referencia: U-Net desde cero

Una U-Net entrenada desde cero con el mismo dataset y el mismo split:

| clase | U-Net | SAM 2 | |
|---|---|---|---|
| background | 0.7768 | **0.8791** | +0.102 |
| track | 0.7365 | **0.8743** | +0.138 |
| kart | 0.6709 | **0.8158** | +0.145 |
| pickup | 0.5013 | **0.5496** | +0.048 |
| bomb | 0.0194 | **0.2180** | +0.199 |
| projectile | 0.0000 | **0.4124** | +0.412 |
| nitro | **0.5352** | 0.4562 | −0.079 |
| **mIoU** | 0.4629 | **0.6008** | **+0.138** |

---

## Experimentos

113 corridas, todas registradas en `experimentos/resultados/`. El `top5` de una corrida es
la media de sus 5 mejores épocas de validación, más estable que el pico.

| experimento | corridas | resultado |
|---|---|---|
| backbone × `dim` × `lr` × exponente de pesos | 24 | `large` +0.020 sobre `small`; `dim` 256 +0.020 sobre 128; `lr` y exponente, sin efecto |
| backbone × resolución × `dim` | 7 | 1024 px no mejora a 448 px; `dim` 512 empeora (−0.012) |
| descongelado del encoder × `encoder-lr` × 4 repeticiones | 74 | el encoder entero con `encoder-lr` 1e-5 es la mejor combinación |
| validación cruzada por circuito | 8 | afinar el encoder mejora en los 3 pliegues, +0.037 de media |

Entre 12 corridas idénticas el `top5` varía con σ 0.011, así que se comparan medias y no
corridas sueltas.

### Cuánto descongelar el encoder

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figuras/descongelado-dark.png">
  <img src="docs/figuras/descongelado-light.png" alt="mIoU según cuánto encoder se descongela y a qué learning rate">
</picture>

| descongelado | parámetros del encoder | `elr` 1e-4 | `elr` 1e-5 | `elr` 1e-6 |
|---|---|:---:|:---:|:---:|
| ninguno | 0 | 0.483 | 0.494 | 0.496 |
| cuello FPN | 0.55 M · 0.3% | 0.464 | 0.487 | 0.485 |
| 3 bloques | 48 M · 23% | 0.470 | 0.483 | 0.470 |
| 15 bloques | 107 M · 50% | 0.500 | 0.509 | 0.505 |
| 28 bloques | 159 M · 75% | 0.528 | 0.548 | 0.501 |
| 48 bloques | 213 M · 100% | 0.542 | **0.564** | 0.509 |

*`top5` medio de 4 repeticiones por celda; `elr` es el learning rate del encoder.*

- Descongelar solo el cuello o los últimos bloques queda por debajo del encoder congelado.
- El encoder entero con `elr` 1e-5 es la mejor combinación: 0.564 frente a 0.491 del congelado.
- Con `elr` 1e-6 los pesos apenas se mueven y el resultado queda cerca del congelado.

### Validación cruzada

Cada par de circuitos pasa una vez a validación, y en cada caso se compara el encoder
congelado con el afinado entero.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figuras/validacion-cruzada-dark.png">
  <img src="docs/figuras/validacion-cruzada-light.png" alt="Mejora de mIoU al afinar el encoder en cada pliegue">
</picture>

| pliegue | circuitos de validación | congelado | afinado | mejora |
|---|---|---|---|---|
| f1 | `lighthouse`, `volcano_island` | 0.491 | 0.564 | +0.073 |
| f2 | `gran_paradiso_island`, `hacienda` | 0.584 | 0.596 | +0.012 |
| f3 | `abyss`, `olivermath` | 0.462 | 0.487 | +0.025 |
| **media** | | | | **+0.037** |

La configuración se eligió con f1, por eso su mejora es la más alta; la media de los tres
pliegues, **+0.037**, es la estimación realista.

### Entrenamiento vs validación

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figuras/entrenamiento-validacion-dark.png">
  <img src="docs/figuras/entrenamiento-validacion-light.png" alt="Pérdida de entrenamiento y de validación del modelo final">
</picture>

La pérdida de entrenamiento sigue bajando y la de validación se estanca desde la época 3.
El mIoU de validación, en cambio, sigue subiendo hasta la época 15, que es la que se guarda:
el modelo se vuelve más confiado en los píxeles que falla, que es lo que la CrossEntropy
penaliza, pero acierta más píxeles.

---

## Limitaciones

- **`projectile` aparece en 2 frames de los 1500**, uno en entrenamiento y uno en
  validación. Su IoU de 0.41 se mide sobre ese único frame y es ruido.
- **`bomb` es la clase más baja** (0.218): el 48.7% de sus píxeles se predice como `pickup`.
- **213 M de parámetros afinados con 1000 imágenes.** La pérdida de validación se estanca
  pronto y la mejora frente al encoder congelado va de +0.012 a +0.073 según los circuitos.
- **No hay conjunto de test independiente.** Todas las cifras son de validación; la
  validación cruzada lo mitiga pero no lo sustituye.

---

## Reproducir

```bash
python train.py --data-root data --backbone large --size 448 --dim 256 \
    --lr 1e-3 --weight-power 0.30 --unfreeze-blocks 48 --unfreeze-neck \
    --encoder-lr 1e-5 --epochs 30 --early-stop 8 --batch-size 8

python experimentos/sweep.py --data-root data --backbones large --sizes 448 --dims 256 \
    --modos none b48 --encoder-lrs 1e-5 --repeticiones 2 --pliegues f2 f3 \
    --epochs 30 --early-stop 8
```

El dataset se organiza en `images/` y `masks/`, emparejados por circuito e identificador.
La máscara se lee sin conversión de color y se redimensiona con **NEAREST**: un
`ToTensor()` la dividiría entre 255 y destruiría la codificación de clases, e interpolarla
inventaría clases inexistentes en los bordes.

### Estructura

- `models.py` — encoder de SAM 2 y cabeza FPN; `save_model` y `load_model`.
- `utils.py` — dataset, métricas y pesos de clase.
- `train.py` — entrenamiento, con `--resume`, descongelado del encoder y `--val-tracks`.
- `predict.py` — inferencia sobre una imagen o una carpeta; descarga el modelo si falta.
- `train.ipynb` — entrenamiento, métricas e imágenes segmentadas.
- `demo.ipynb` — demo en Colab.
- `model.th` — modelo entrenado, 861 MB, en *Releases*.
- `class_weights.json` — conteo de píxeles por clase del conjunto de entrenamiento.
- `ejemplos/` — imágenes para probar sin descargar el dataset.
- `experimentos/` — barrido multi-GPU, validación cruzada y resultados de las 113 corridas.
- `servidor/` — scripts para entrenar en un servidor GPU remoto con Docker; el host se
  indica con `STK_SERVER=usuario@host`.
- `docs/` — figuras del README y el script que las genera.

Entrenado en GPUs NVIDIA H200 con Docker (torch 2.13, CUDA). Probado en macOS con torch 2.14
sobre CPU. Depende de `torch`, `torchvision`, `numpy`, `Pillow` y `sam2`.
